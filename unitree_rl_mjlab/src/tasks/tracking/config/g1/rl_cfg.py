"""RL configuration for Unitree G1 tracking task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class EpisodeLengthCurriculumCfg:
  """Grow the episode length once the agent tracks well AND survives the cap.

  Training always restarts the gauntlet from ``start_s`` (every run re-runs it
  from scratch, even when weights are resumed). The episode cap is increased by
  ``increment_s`` only when BOTH hold over the last 100 terminated episodes:

  - the mean episode length reaches ``trigger_ratio`` of the current cap, and
  - the mean reward per step reaches ``min_mean_reward_per_step``.

  ``min_mean_reward_per_step`` keeps "survival" (time-out) from being enough:
  the agent must actually track well (the tracking rewards are exp-error terms,
  perfect tracking ~5.0/step) before being asked to track more of the motion.
  Growth stops once the cap covers the full reference motion duration.
  """

  enable: bool = True
  start_s: float = 10.0
  increment_s: float = 5.0
  trigger_ratio: float = 0.95
  # Calibrated 2026-08-10 from live runs: rewards are scaled by step_dt (0.02),
  # so the perfect-tracking ceiling is ~5.0 * 0.02 = 0.10/step. Two runs both
  # plateaued around 0.063-0.064/step (63-64% of the ceiling): a 5000+ iter run
  # at mean reward 32 / length 500 = 0.064. The policy CANNOT exceed ~0.064 on
  # this reference, so a gate at 0.07-0.08 would stall the curriculum forever.
  # 0.05 (=~78% of the plateau) still excludes pure survival while letting the
  # cap grow once tracking is at the policy's achievable level.
  min_mean_reward_per_step: float = 0.05


@dataclass
class ThresholdCurriculumCfg:
  """Progressively tighten the termination thresholds as tracking improves.

  Each time the mean reward/step rises ``reward_step_increment`` above the
  level at the last tightening (while the agent is also surviving near the
  current cap), the three termination thresholds (anchor_pos / anchor_ori /
  ee_body_pos) are multiplied by ``tighten_factor``, down to the per-term
  floors. Tighter thresholds force the policy to track more precisely, so the
  reward keeps rising with training instead of plateauing. Set ``enable=False``
  to disable.
  """

  enable: bool = True
  # 0.003 (was 0.005): tighten on smaller reward/step gains so the precision
  # phase (after full-motion coverage) keeps pushing the policy harder.
  reward_step_increment: float = 0.003
  tighten_factor: float = 0.9
  # Floors raised (2026-08-11): the curriculum over-tightened anchor_pos down to
  # ~0.10 below what the robot achieves (83% anchor_pos terminations). These
  # floors stop tightening near the robot's real capability.
  min_anchor_pos: float = 0.25
  min_anchor_ori: float = 0.3
  min_ee_body_pos: float = 0.35


@dataclass
class G1TrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
  # Master switch for the standing-intro/ending free-connection strategy
  # (free-connection windows + warmup + episode-length/threshold curricula).
  # Leave True for the dance-with-standing model; set False to train any other
  # model with standard tracking (e.g. --agent.free-connect-style=False).
  free_connect_style: bool = True
  episode_length_curriculum: EpisodeLengthCurriculumCfg = field(
    default_factory=EpisodeLengthCurriculumCfg
  )
  threshold_curriculum: ThresholdCurriculumCfg = field(
    default_factory=ThresholdCurriculumCfg
  )


def unitree_g1_tracking_ppo_runner_cfg() -> G1TrackingRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return G1TrackingRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.15,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
  )