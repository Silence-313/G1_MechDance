import os
import statistics
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self._free_connect_style = self.cfg.get("free_connect_style", True)
    if not self._free_connect_style:
      self._disable_free_connection_env()
    self._setup_episode_length_curriculum()
    self._setup_threshold_curriculum()
    self._setup_curriculum_hooks()

  def _disable_free_connection_env(self) -> None:
    """Zero the free-connection windows and warmup when free_connect_style=False.

    The command manager does NOT deepcopy the command cfg, so mutating
    ``env.cfg.commands["motion"]`` affects the already-built motion command.
    The termination manager deepcopies its cfg, so warmup is zeroed on
    ``env.termination_manager.cfg`` directly (like the threshold curriculum).
    """
    env = self.env.unwrapped
    motion_cfg = env.cfg.commands["motion"]
    motion_cfg.free_connection_before = 0
    motion_cfg.free_connection_after = 0
    tm = env.termination_manager.cfg
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
      if name in tm and "warmup_frames" in tm[name].params:
        tm[name].params["warmup_frames"] = 0
    print(
      "[FreeConnectStyle] disabled: standard tracking (no free windows, "
      "no warmup, no curricula)."
    )

  def _setup_episode_length_curriculum(self) -> None:
    """Read the episode-length curriculum config.

    The episode cap is always reset to ``start_s`` here, so every training run
    (fresh or resumed) re-runs the gauntlet from scratch. Play mode sets
    ``episode_length_s`` to a huge sentinel value, which is left untouched.
    """
    curriculum = self.cfg.get("episode_length_curriculum", {})
    self._curriculum_enabled = curriculum.get("enable", True)
    if not self._free_connect_style:
      self._curriculum_enabled = False
    self._curriculum_start_s = curriculum.get("start_s", 10.0)
    self._curriculum_increment_s = curriculum.get("increment_s", 5.0)
    self._curriculum_trigger_ratio = curriculum.get("trigger_ratio", 0.95)
    self._curriculum_min_reward_per_step = curriculum.get(
      "min_mean_reward_per_step", 0.05
    )
    if not self._curriculum_enabled:
      return
    env = self.env.unwrapped
    if env.cfg.episode_length_s < 1e8:  # not play mode
      env.cfg.episode_length_s = self._curriculum_start_s
    print(
      "[EpisodeLengthCurriculum] enabled: start "
      f"{self._curriculum_start_s}s, +{self._curriculum_increment_s}s when "
      f"mean_ep_len >= {self._curriculum_trigger_ratio:.0%} of cap AND "
      f"mean_reward/step >= {self._curriculum_min_reward_per_step}, "
      "up to the full motion"
    )

  def _setup_threshold_curriculum(self) -> None:
    """Read the termination-threshold curriculum config."""
    curriculum = self.cfg.get("threshold_curriculum", {})
    self._threshold_enabled = curriculum.get("enable", True)
    if not self._free_connect_style:
      self._threshold_enabled = False
    self._threshold_reward_step_increment = curriculum.get(
      "reward_step_increment", 0.003
    )
    self._threshold_tighten_factor = curriculum.get("tighten_factor", 0.9)
    self._threshold_min = {
      "anchor_pos": curriculum.get("min_anchor_pos", 0.25),
      "anchor_ori": curriculum.get("min_anchor_ori", 0.3),
      "ee_body_pos": curriculum.get("min_ee_body_pos", 0.35),
    }
    # Start tightening only once the policy tracks at the episode-length reward
    # gate level, so weak early policies aren't pushed over the edge.
    self._last_tighten_reward = self._curriculum_min_reward_per_step
    if not self._threshold_enabled:
      return
    print(
      "[ThresholdCurriculum] enabled: tighten termination thresholds x"
      f"{self._threshold_tighten_factor} when reward/step rises "
      f"{self._threshold_reward_step_increment} above the last-tighten level"
    )

  def _setup_curriculum_hooks(self) -> None:
    """Wrap ``logger.log`` so both curricula run once per training iteration.

    Only runs inside ``learn()`` (play never calls ``logger.log``).
    """
    if not (self._curriculum_enabled or self._threshold_enabled):
      return
    orig_log = self.logger.log

    def log_with_curriculum(*args, **kwargs):
      self._maybe_grow_episode_length()
      self._maybe_tighten_thresholds()
      return orig_log(*args, **kwargs)

    self.logger.log = log_with_curriculum

  def _maybe_grow_episode_length(self) -> None:
    """Grow episode_length_s when the agent survives AND tracks well.

    ``self.logger.lenbuffer`` / ``rewbuffer`` hold the length and cumulative
    reward of the last 100 terminated episodes. Growth requires both:

    - mean episode length >= ``trigger_ratio`` of the current cap, and
    - mean reward per step >= ``min_mean_reward_per_step``.

    This keeps mere survival (time-out) from being enough — the tracking
    rewards are exp-error terms (perfect tracking ~5.0/step), so the reward
    gate only opens once the agent actually imitates the motion. The cap is
    clamped to the full reference motion duration. ``env.max_episode_length``
    is a property derived from ``episode_length_s``, so the new cap takes
    effect immediately.
    """
    if not self._curriculum_enabled:
      return
    env = self.env.unwrapped
    try:
      motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
      motion = motion_term.motion
    except Exception:
      return
    full_s = motion.time_step_total * env.step_dt
    current_s = env.cfg.episode_length_s
    if current_s >= full_s:
      return
    if not self.logger.lenbuffer:
      return
    mean_len = statistics.mean(self.logger.lenbuffer)
    mean_rew_per_step = sum(self.logger.rewbuffer) / sum(self.logger.lenbuffer)
    if (
      mean_len >= self._curriculum_trigger_ratio * env.max_episode_length
      and mean_rew_per_step >= self._curriculum_min_reward_per_step
    ):
      new_s = min(current_s + self._curriculum_increment_s, full_s)
      env.cfg.episode_length_s = new_s
      print(
        f"[EpisodeLengthCurriculum] mean_ep_len={mean_len:.0f}/"
        f"{env.max_episode_length}, mean_rew/step={mean_rew_per_step:.2f} "
        f"-> episode_length_s {current_s:.1f}s -> {new_s:.1f}s "
        f"(full {full_s:.1f}s)"
      )

  def _maybe_tighten_thresholds(self) -> None:
    """Tighten termination thresholds as the tracking reward improves.

    When mean reward/step has risen ``reward_step_increment`` above the level
    at the last tightening (and the agent is surviving near the current cap),
    multiply the anchor_pos / anchor_ori / ee_body_pos termination thresholds
    by ``tighten_factor``, clamped to the configured floors. This forces the
    policy to keep tracking more precisely, so reward rises with training
    instead of plateauing.

    The termination manager holds a deepcopy of the cfg, so thresholds are
    updated on ``env.termination_manager.cfg`` (which ``compute()`` reads every
    step), NOT on ``env.cfg.terminations``.
    """
    if not self._threshold_enabled:
      return
    env = self.env.unwrapped
    if not self.logger.lenbuffer:
      return
    mean_rew_per_step = sum(self.logger.rewbuffer) / sum(self.logger.lenbuffer)
    if mean_rew_per_step < self._last_tighten_reward + self._threshold_reward_step_increment:
      return
    # Only tighten while comfortably surviving, so it doesn't push a struggling
    # policy over the edge.
    mean_len = statistics.mean(self.logger.lenbuffer)
    if mean_len < self._curriculum_trigger_ratio * env.max_episode_length:
      return
    tm = env.termination_manager.cfg
    changes = {}
    for name, min_val in self._threshold_min.items():
      term = tm[name].params
      old = term["threshold"]
      new = max(old * self._threshold_tighten_factor, min_val)
      term["threshold"] = new
      changes[name] = f"{old:.2f}->{new:.2f}"
    self._last_tighten_reward = mean_rew_per_step
    print(
      f"[ThresholdCurriculum] reward/step={mean_rew_per_step:.3f} "
      f"(survival {mean_len:.0f}/{env.max_episode_length}) -> thresholds {changes}"
    )

  def export_motion_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    self.export_motion_policy_to_onnx(policy_path, filename)
    self.export_policy_to_onnx(policy_path, "policy.onnx")
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
      }
    )
    attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
