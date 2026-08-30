from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _free_aware(command: MotionCommand, reward: torch.Tensor) -> torch.Tensor:
  """Zero the tracking reward during free-connection windows.

  In the standing intro / ending segments the agent is free to find its own
  transition into/out of the dance, so no tracking reward is imposed there.
  """
  free = command.free_connection_mask()
  if free is None:
    return reward
  return torch.where(free, torch.zeros_like(reward), reward)


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return _free_aware(command, torch.exp(-error / std**2))


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return _free_aware(command, torch.exp(-error / std**2))


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _free_aware(command, torch.exp(-error.mean(-1) / std**2))


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return _free_aware(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _free_aware(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _free_aware(command, torch.exp(-error.mean(-1) / std**2))


def base_upright_free_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
  pos_std: float = 0.3,
) -> torch.Tensor:
  """Reward staying upright AND near the reference position during free windows.

  In the free windows the tracking rewards are zeroed and terminations masked,
  so without any signal the agent can tip over (NaN) or wander away in a weird
  balancing pose. This rewards the base for being upright (gravity aligned with
  its -z) AND staying near the reference anchor in the horizontal plane. The
  horizontal-only constraint lets the robot crouch/transition freely while
  keeping it planted instead of drifting sideways.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  free = command.free_connection_mask()
  if free is None:
    return torch.zeros(env.num_envs, device=env.device)
  asset = env.scene["robot"]
  base_pos = asset.data.body_link_pos_w[:, 0]
  base_quat = asset.data.body_link_quat_w[:, 0]
  gravity = asset.data.gravity_vec_w
  upright = -quat_apply_inverse(base_quat, gravity)[:, 2]
  upright_r = torch.exp(-torch.square(1.0 - upright) / std**2)
  # horizontal distance to the reference anchor (vertical/crouch is free)
  horiz_err = torch.norm(base_pos[:, :2] - command.anchor_pos_w[:, :2], dim=-1)
  pos_r = torch.exp(-torch.square(horiz_err) / pos_std**2)
  reward = upright_r * pos_r
  return torch.where(free, reward, torch.zeros_like(reward))


def free_transition_guide_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """During free windows, guide the transition into/out of the dance.

  Instead of trying to balance standing still (which made the agent fall), the
  free windows guide it:
  - start window (first ``free_connection_before`` frames): gradually crouch
    into the SQUAT (dance-start pose, reference frame ``free_connection_before``),
  - end window (last ``free_connection_after`` frames): return to STANDING
    (reference frame ``time_step_total - 1``).

  Replicates the relative-body tracking error computation but uses those target
  reference frames. 0 elsewhere.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  before = command.cfg.free_connection_before
  after = command.cfg.free_connection_after
  if before <= 0 and after <= 0:
    return torch.zeros(env.num_envs, device=env.device)
  total = command.motion.time_step_total
  idx = _get_body_indexes(command, body_names)
  device = env.device
  B = env.num_envs
  J = len(command.cfg.body_names)
  origins = env.scene.env_origins[:, None, :]  # (B,1,3)

  def rel_error(frame: int) -> torch.Tensor:
    """Per-env body-position error vs the reference at ``frame`` (same relative
    transform as the base command's _update_command)."""
    target_body_w = torch.as_tensor(command.motion.body_pos_w[frame], device=device)  # (J,3)
    target_anchor_w = torch.as_tensor(
      command.motion.body_pos_w[frame, command.motion_anchor_body_index], device=device
    )  # (3,)
    target_anchor_q = torch.as_tensor(
      command.motion.body_quat_w[frame, command.motion_anchor_body_index], device=device
    )  # (4,)
    target_body_w_e = (target_body_w[None] + origins).expand(B, J, 3)
    anchor_pos_repeat = target_anchor_w[None, None].expand(B, J, 3)
    anchor_quat_repeat = target_anchor_q[None, None].expand(B, J, 4)
    robot_anchor_pos_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, J, 1)
    robot_anchor_quat_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, J, 1)
    delta_pos = robot_anchor_pos_repeat.clone()
    delta_pos[..., 2] = anchor_pos_repeat[..., 2]
    delta_ori = yaw_quat(
      quat_mul(robot_anchor_quat_repeat, quat_inv(anchor_quat_repeat))
    )
    rel = delta_pos + quat_apply(delta_ori, target_body_w_e - anchor_pos_repeat)
    return torch.norm(rel[:, idx] - command.robot_body_pos_w[:, idx], dim=-1)

  err_start = rel_error(before) if before > 0 else None   # -> squat
  err_end = rel_error(total - 1) if after > 0 else None    # -> standing
  # Linear guide (not exp): exp saturates to ~0 when the robot is far from the
  # target, so it provided no gradient to pull it into the squat/standing.
  # Linear gives a usable gradient across the whole range err < std.
  zeros = torch.zeros(B, device=device)
  reward = zeros
  if err_start is not None:
    r_start = torch.clamp(1.0 - err_start.mean(-1) / std, min=0.0)
    reward = torch.where(command.time_steps < before, r_start, reward)
  if err_end is not None:
    r_end = torch.clamp(1.0 - err_end.mean(-1) / std, min=0.0)
    reward = torch.where(command.time_steps >= total - after, r_end, reward)
  return reward


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
