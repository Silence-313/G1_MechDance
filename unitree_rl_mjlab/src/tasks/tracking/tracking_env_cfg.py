"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from src.tasks.tracking.mdp.curriculum_commands import MotionCommandCfg
from src.tasks.tracking.mdp.rewards import (
  base_upright_free_reward,
  free_transition_guide_reward,
)
from src.tasks.tracking.mdp.terminations import (
  bad_anchor_ori,
  bad_anchor_pos_z_only,
  bad_motion_body_pos_z_only,
  base_fallen_free,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.tracking.mdp as mdp

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base tracking task configuration."""

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
      # FREE-connection windows DISABLED (2026-08-14): both set to 0. The
      # reference motion (17fe0adfa...) already has a tracked standing intro
      # and ending, so the robot tracks them directly instead of learning its
      # own standing<->squat transitions. With both 0, `free_connection_mask()`
      # returns None, which makes base_upright_free / free_transition_guide
      # rewards and base_fallen_free termination all no-ops, and removes the
      # termination masking. (Previously 200/100; the free ending left the
      # robot unguided after the dance and it fell.)
      free_connection_before=0,
      free_connection_after=0,
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    # Root-orientation weight (0.5 -> 2.0 -> 4.0): the body tracking rewards are
    # yaw-invariant (they align to the robot's heading before comparing), so the
    # ONLY signal for the 360° turn is this term. The robot still skips the fast
    # 360° spin; 4.0 makes the turn worth the risk.
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=4.0,
      params={"command_name": "motion", "std": 0.6},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.5},
    ),
    # Dedicated ankle-position reward: `motion_body_pos` averages over all 14
    # bodies, so the 2 ankles' lift/step error (~0.2 m) is diluted to ~0.1
    # effective weight and the policy keeps both feet planted. This term gives
    # the feet their own (un-diluted) signal so lifting/stepping pays off.
    # std tightened 0.3 -> 0.12 (2026-08-14): the exp reward saturates near
    # zero error, so the policy undershot both small AND large leg motions
    # ("close enough" = reward near 1). The smaller std keeps the gradient
    # alive down to ~2 cm, forcing full-range foot/leg tracking.
    "motion_body_pos_feet": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=3.5,
      params={
        "command_name": "motion",
        "std": 0.12,
        "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
      },
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.6},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # Reduced -0.1 -> -0.03 so the fast 360° spin (which needs rapid joint
    # changes) isn't penalized into being skipped.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-3e-2),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # Free-window stability: keep the base upright AND near the reference
    # position during free-connection frames (0 elsewhere), so the robot
    # doesn't tip over (NaN) or wander in a weird balancing pose. Crouching is
    # fine (base stays upright, horizontal-only position constraint). Weight
    # raised 1.0 -> 2.0 (2026-08-11) because the agent struggled to learn to
    # stand still in the free window.
    "base_upright_free": RewardTermCfg(
      func=base_upright_free_reward,
      weight=2.0,
      params={"command_name": "motion", "std": 0.5, "pos_std": 0.3},
    ),
    # Free-window transition guide: start window -> crouch into the squat;
    # end window -> return to standing. Guides the agent through the
    # connections instead of making it balance standing still (which fell).
    "free_transition_guide": RewardTermCfg(
      func=free_transition_guide_reward,
      weight=2.0,
      params={"command_name": "motion", "std": 1.0},
    ),
  }

  ##
  # Terminations
  ##

  # Thresholds tightened from the previous 0.50/0.8 to force precise tracking
  # (survival no longer suffices; the episode-length curriculum additionally
  # gates growth on reward). NOTE: the reference motion contains SQUATS, which
# legitimately lower the torso and wrists a lot. The thresholds measure
  # deviation FROM the reference, so a correctly-tracked squat stays inside
  # them, but the wrist-z (ee_body_pos) threshold is kept looser than the
  # torso ones so fast/deep squats don't false-terminate before the policy
  # learns them. Revisit if squat segments terminate spuriously.
  # warmup_frames is REFERENCE-FRAME based (not episode-step based): termination
  # is masked while the current reference frame < warmup_frames. The reference
  # is now standing intro (100) + transition into the squat (100) + the dance
  # starting from a squat, so warmup_frames=300 covers the intro, the squat, and
  # ~100 frames into the dance — the policy is protected through the hard
  # "stand -> squat" segment instead of the old 200-step (episode-based) window
  # that expired before the squat. In RSI "uniform" starts past frame 300 there
  # is no warm-up (the intro isn't being replayed), which is correct.
  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=bad_anchor_pos_z_only,
      params={
        "command_name": "motion",
        # 0.35 (was 0.25): relaxed so the now-bigger movements (turn, kicks)
        # don't kill the robot on torso z deviation.
        "threshold": 0.35,
        "warmup_frames": 300,
      },
    ),
    "anchor_ori": TerminationTermCfg(
      func=bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        # 0.5 rad (was 0.6): tightened again (2026-08-10) to force finer
        # tracking once the policy reached the full motion at reward/step
        # ~0.065. Watch orientation: if the policy re-climbs the gauntlet and
        # then orientation terminations spike, ease back toward 0.6.
        "threshold": 0.5,
        "warmup_frames": 300,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        # 0.45 (was 0.30): relaxed so the big dance movements (turn, kicks, arm
        # swings) don't kill the robot on wrist/ankle z deviation.
        "threshold": 0.45,
        "body_names": (),  # Set per-robot.
        "warmup_frames": 300,
      },
    ),
    # Free-window fall detection: resets a robot that tips over during a free
    # window (where tracking terminations are masked), preventing it from lying
    # in a degenerate contact state that can produce NaN obs.
    "base_fallen_free": TerminationTermCfg(
      func=base_fallen_free,
      params={"command_name": "motion", "min_height": 0.3, "max_tilt": 1.2},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=512),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Raised (2026-08-11): during the free-connection windows the robot
      # explores freely and can fall into complex contact states, overflowing
      # the previous nconmax=35 / njmax=250 ("nefc overflow" -> NaN obs).
      nconmax=64,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
