"""Curriculum-enabled motion command for tracking tasks.

The base mjlab motion command implements Reference State Initialization (RSI):
on reset the robot is placed at a random reference-motion frame. With an
adaptive/uniform sampling mode the robot can be dropped into a complex mid-dance
pose and falls within a few steps, so the policy never gathers tracking
experience.

This module subclasses the mjlab motion command (minimal modification, no
copying of the whole command system) to add:

1. A **staged RSI curriculum**: during early training the robot always starts
   from the easiest reference frames, then progressively switches to harder
   random sampling once the policy can survive:

   - ``step < curriculum_start_iters``   -> sampling_mode ``"start"`` (frame 0)
   - ``step < curriculum_uniform_iters`` -> sampling_mode ``"uniform"``
   - ``step >= curriculum_uniform_iters``-> configured ``sampling_mode`` (adaptive)

2. A **reset stale-state fix**: on an initial reset the env does not run
   ``command_manager.compute()``, so ``body_pos_relative_w`` /
   ``body_quat_relative_w`` stay zero and the first termination check can be
   triggered spuriously. ``reset()`` refreshes the relative state right after
   the reference state is written, without changing the normal step order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand as _BaseMotionCommand,
  MotionCommandCfg as _BaseMotionCommandCfg,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionCommandCfg(_BaseMotionCommandCfg):
  """Motion command config with a staged RSI curriculum.

  ``curriculum_start_iters`` / ``curriculum_uniform_iters`` are env-step
  thresholds (env steps = PPO iteration * num_steps_per_env). Leaving both at
  their defaults preserves the original behavior exactly.
  """

  curriculum_start_iters: int = 0
  """Env steps during which sampling_mode is forced to ``"start"`` (frame 0)."""
  curriculum_uniform_iters: int = 0
  """Env steps during which sampling_mode is forced to ``"uniform"``."""
  force_start_mode: bool = False
  """Temporary A/B switch: if True, always force ``"start"`` (frame 0) and
  disable the staged curriculum entirely."""
  use_adaptive_sampling: bool = False
  """If True, release to configured ``sampling_mode`` (typically adaptive)
  after ``curriculum_uniform_iters``. If False (default), stays in uniform
  mode forever — adaptive sampling is permanently disabled."""

  free_connection_before: int = 0
  """Reference frames at the START where tracking is FREE: the agent may move
  however it likes (tracking rewards zeroed, terminations masked) so it can
  learn its own standing -> squat transition. Should equal the reference's
  standing-before segment length."""
  free_connection_after: int = 0
  """Reference frames at the END where tracking is FREE, so the agent learns
  its own dance -> standing transition. Should equal the reference's
  standing-after segment length."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


class MotionCommand(_BaseMotionCommand):
  cfg: MotionCommandCfg

  def free_connection_mask(self) -> torch.Tensor | None:
    """Per-env bool: True where the current reference frame is in a FREE window.

    ``free_connection_before`` = first N reference frames (the standing intro
    where the agent learns its own way into the squat); ``free_connection_after``
    = last M reference frames (the ending where it learns its own way back to
    standing). Returns None when both are 0 (no free windows).
    """
    before = self.cfg.free_connection_before
    after = self.cfg.free_connection_after
    if before <= 0 and after <= 0:
      return None
    total = self.motion.time_step_total
    return (self.time_steps < before) | (self.time_steps >= total - after)

  def reset(self, env_ids) -> dict[str, float]:
    """Resample the command and immediately refresh the relative body state.

    The base reset writes the reference robot state (RSI) but not the
    relative-body buffers. On an *initial* reset no ``command_manager.compute()``
    runs afterwards, so those buffers stay zero and the first termination check
    sees a huge fake error. Refreshing here fixes that without touching the
    normal step order.
    """
    extras = super().reset(env_ids)
    self._refresh_relative_state()
    return extras

  def _resample_command(self, env_ids) -> None:
    forced_mode = self._curriculum_mode()
    if forced_mode is not None:
      # Temporarily force the curriculum sampling mode, then hand over.
      original_mode = self.cfg.sampling_mode
      self.cfg.sampling_mode = forced_mode
      try:
        super()._resample_command(env_ids)
      finally:
        self.cfg.sampling_mode = original_mode
    else:
      super()._resample_command(env_ids)

  def _curriculum_mode(self) -> str | None:
    if self.cfg.force_start_mode:
      mode = "start"
    else:
      step = self._env.common_step_counter
      if step < self.cfg.curriculum_start_iters:
        mode = "start"
      elif step < self.cfg.curriculum_uniform_iters:
        mode = "uniform"
      elif self.cfg.use_adaptive_sampling:
        return None  # release to configured sampling_mode (adaptive)
      else:
        mode = "uniform"  # adaptive disabled, stay uniform forever

    # Log when the curriculum stage changes.
    last = getattr(self, "_last_logged_mode", None)
    if mode != last:
      self._last_logged_mode = mode
      print(
        f"[RSI Curriculum] step={self._env.common_step_counter}, "
        f"mode={mode}, "
        f"use_adaptive={self.cfg.use_adaptive_sampling}"
      )
    return mode

  def _refresh_relative_state(self) -> None:
    """Recompute body_pos_relative_w / body_quat_relative_w from current state.

    Mirrors the relative-pose block of the base ``_update_command`` (without the
    ``time_steps += 1`` / adaptive bin bookkeeping) so it can be called right
    after a reset.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )
