from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  warmup_frames: int = 0,
) -> torch.Tensor:
  """Anchor z-deviation termination with an optional warm-up window.

  Termination is masked while the current reference frame index
  (``command.time_steps``) is below ``warmup_frames``. This is reference-frame
  based (not episode-step based), so it correctly protects the early reference
  frames — e.g. a standing intro + transition into a deep squat — regardless of
  where the episode started. With RSI "start" mode (frame 0) the robot is
  protected through the intro and into the squat; with "uniform" starts past
  ``warmup_frames`` there is no warm-up (the hard intro isn't being replayed).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  terminated = (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )
  if warmup_frames > 0:
    terminated = terminated & (command.time_steps >= warmup_frames)
  free = command.free_connection_mask()
  if free is not None:
    terminated = terminated & ~free
  return terminated


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
  warmup_frames: int = 0,
) -> torch.Tensor:
  """Anchor orientation-deviation termination with a reference-frame warm-up."""
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  terminated = (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold
  if warmup_frames > 0:
    terminated = terminated & (command.time_steps >= warmup_frames)
  free = command.free_connection_mask()
  if free is not None:
    terminated = terminated & ~free
  return terminated


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  warmup_frames: int = 0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  terminated = torch.any(error > threshold, dim=-1)
  if warmup_frames > 0:
    terminated = terminated & (command.time_steps >= warmup_frames)
  free = command.free_connection_mask()
  if free is not None:
    terminated = terminated & ~free
  return terminated


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
  warmup_frames: int = 0,
) -> torch.Tensor:
  """Body z-deviation termination with a reference-frame warm-up."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  terminated = torch.any(error > threshold, dim=-1)
  if warmup_frames > 0:
    terminated = terminated & (command.time_steps >= warmup_frames)
  free = command.free_connection_mask()
  if free is not None:
    terminated = terminated & ~free
  return terminated


def base_fallen_free(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_height: float = 0.3,
  max_tilt: float = 1.2,
) -> torch.Tensor:
  """Reset a robot that falls during a free-connection window.

  The tracking terminations are masked in the free windows, so a robot that
  tips over there would otherwise lie in a degenerate contact state and can
  produce NaN in the sim. This uses ABSOLUTE base height / uprightness (NOT
  reference-relative) and only fires during free frames. ``upright`` = -z of
  gravity in the base frame (~1 when upright; a squat keeps the base upright).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  free = command.free_connection_mask()
  if free is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  asset: Entity = env.scene["robot"]
  base_pos = asset.data.body_link_pos_w[:, 0]
  base_quat = asset.data.body_link_quat_w[:, 0]
  gravity = asset.data.gravity_vec_w
  upright = -quat_apply_inverse(base_quat, gravity)[:, 2]
  fallen = (base_pos[:, 2] < min_height) | (upright < float(math.cos(max_tilt)))
  return fallen & free
