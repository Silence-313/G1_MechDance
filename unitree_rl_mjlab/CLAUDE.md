# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Mjlab is a reinforcement learning framework for legged robot locomotion and sim-to-real deployment. Built on [mjlab](https://github.com/mujocolab/mjlab) with MuJoCo physics, it supports Unitree Go2, A2, As2, G1 (29-dof and 23-dof variants), R1, H1_2, and H2 robots.

Two RL task families:
- **Velocity tracking**: Locomotion policy following twist commands (linear/angular velocity)
- **Motion tracking (imitation)**: Replay reference motion clips (reimplementation of [BeyondMimic](https://beyondmimic.github.io/))

## Build / Install

Requires Python 3.11. Python dependencies are `mjlab==1.2.0` and `mujoco-warp==3.5.0`. Install with:

```bash
pip install -e .
```

For C++ deployment builds, system dependencies are also needed:

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

C++ deployment also requires CMake, unitree_sdk2, cyclonedds, and ONNX Runtime (pre-bundled in `deploy/thirdparty/` for both aarch64 and x64).

There are no automated tests in this repository.

## Common Commands

**Train:**
```bash
# Single GPU
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# Multi-GPU
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096

# Motion imitation
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance.npz --env.scene.num-envs=4096

# Resume from checkpoint
python scripts/train.py Unitree-G1-Flat --agent.resume --agent.load_run=<run_dir>

# Record training videos (every 2000 steps)
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096 --video

# Enable NaN guard for debugging simulation instability
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096 --enable-nan-guard
```

**Play (visualize):**
```bash
# Trained policy (local checkpoint)
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt

# Trained policy (from WandB)
python scripts/play.py Unitree-G1-Flat --wandb_run_path=<entity/project/run_id>

# Record video of trained policy
python scripts/play.py Unitree-G1-Flat --checkpoint_file=<path> --video

# Zero-action / random agent for debugging
python scripts/play.py Unitree-G1-Flat --agent=zero
python scripts/play.py Unitree-G1-Flat --agent=random

# Disable terminations (useful for viewing motions with dummy agents)
python scripts/play.py Unitree-G1-Flat --agent=zero --no-terminations

# Force a specific viewer backend
python scripts/play.py Unitree-G1-Flat --agent=zero --viewer=viser
```

**Motion preprocessing for imitation:**
```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance.csv --output-name dance.npz --input-fps 30 --output-fps 50 --robot g1
```

**List all registered tasks:**
```bash
python scripts/list_envs.py
```

**Visualize terrain:**
```bash
python scripts/visualize_terrain.py
```

**Simulation deployment testing (before real robot):**
```bash
# Build the simulator
cd simulate && mkdir -p build && cd build && cmake .. && make -j8

# Launch simulator (requires gamepad connected)
./simulate/build/unitree_mujoco

# Build and run the control program (in another terminal)
cd deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make
./g1_ctrl --network=lo
```
Select the robot in `simulate/config.yaml` (supports g1, g1_23dof, h1_2, go2, a2).

**Real robot deployment:**
```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0  # Ethernet interface to robot (check with ifconfig)
```
Place `policy.onnx` and `policy.onnx.data` into `deploy/robots/g1/config/policy/velocity/v0/exported` before building.

## Architecture

### Task Registration

Tasks are registered via `register_mjlab_task()` in `src/tasks/*/config/<robot>/__init__.py`. `src/tasks/__init__.py` auto-discovers them at import time using mjlab's `import_packages()`. Each task has three components:

1. **env_cfg** (`*_env_cfg()` factory in `env_cfgs.py`): Returns `ManagerBasedRlEnvCfg` — scene, observations (actor/critic groups), actions, commands, events (domain randomization), rewards, terminations, curriculum.
2. **rl_cfg** (`*_ppo_runner_cfg()` in `rl_cfg.py`): PPO hyperparameters, network architecture (hidden dims, activation), experiment name.
3. **runner_cls** (optional, in `src/tasks/*/rl/runner.py`): Subclass of `MjlabOnPolicyRunner` with custom ONNX export. Velocity tasks export a standard policy ONNX; tracking tasks bundle motion reference data into the ONNX model.

Naming convention: `Unitree-<Robot>-<Terrain>` for velocity, `Unitree-<Robot>-Tracking[-No-State-Estimation]` for imitation.

Terrain types for velocity tasks: `Flat` (infinite plane, no height scan, lighter sim params) and `Rough` (procedurally generated terrain with curriculum, height scan observations). Flat env cfg inherits from Rough then removes terrain complexity.

"No-State-Estimation" means the policy receives no explicit base velocity estimate — it must infer motion from joint states, IMU, and command history alone. This matches real-robot conditions where velocity estimation is unreliable. The critic still gets privileged base velocity during training.

### Actor/Critic Observation Asymmetry

The actor and critic receive different observation sets. The critic gets **privileged information** that the actor does not:
- `base_lin_vel` (IMU linear velocity)
- `foot_height`, `foot_air_time`, `foot_contact`, `foot_contact_forces`
- Uncorrupted `height_scan` (no noise applied)

The actor's observations have noise applied and lack these privileged terms. This is a standard sim-to-real pattern: the critic can use privileged simulation state during training, while the actor only sees realistic sensor data available on the real robot.

### Config Factory Pattern

`velocity_env_cfg.py` and `tracking_env_cfg.py` each define a `make_*_env_cfg()` factory returning a base config with sensible defaults. Robot-specific configs (e.g., `src/tasks/velocity/config/g1/env_cfgs.py`) call the factory then customize: action scales, joint-specific reward parameters, contact sensor setup, collision filter geom names, etc. All factories accept a `play` boolean — when True, they disable domain randomization, set infinite episode length, and add terrain randomization for evaluation.

### Robot Assets

`src/assets/robots/<robot>/` contains per-robot Python constants (`*_constants.py`) defining:
- MJCF XML specs and mesh assets
- Actuator configs (stiffness, damping, effort limits, armature) per motor type
- Initial state keyframes (home pose, knees-bent pose)
- Collision filter configurations

All robots export a `get_<robot>_robot_cfg()` -> `EntityCfg` function used by task configs.

### MDP Modules

Both task families have custom MDP implementations in `src/tasks/*/mdp/` (observations, rewards, terminations, commands). These are **separate from** mjlab's built-in MDP modules and are imported under the same `mdp` alias, shadowing mjlab's defaults where needed.

### Training Flow

`scripts/train.py` uses `tyro` for CLI parsing. The first positional arg selects a registered task. It loads env/rl configs, optionally injects a motion file for tracking tasks, creates a `ManagerBasedRlEnv`, wraps it in `RslRlVecEnvWrapper`, and runs PPO via the runner. Multi-GPU training uses `torchrunx`. Checkpoints and ONNX exports are saved to `logs/rsl_rl/<experiment_name>/<datetime>/`.

### Deployment (C++)

`deploy/robots/<robot>/` contains a C++ control program for each supported robot (a2, g1, g1_23dof, go2, h1_2, r1). Note that As2 and H2 have training support but no deploy directory yet.

Deploy architecture:
- **FSM pattern**: `CtrlFSM` manages states. Shared states (`Passive`, `FixStand`, `RLBase`) live in `deploy/include/FSM/`. Robot-specific states (e.g., `Mimic` for G1/G1_23dof) live in `deploy/robots/<robot>/include/`.
- **ONNX Runtime** for policy inference
- **unitree_sdk2** via CycloneDDS for robot communication

`simulate/` wraps [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) for simulation-based deployment testing before real-robot deployment. Configure the target robot in `simulate/config.yaml`.

### Logging and Export

During training, the runner's `save()` method is called periodically (every `save_interval` iterations, default 100). Both `VelocityOnPolicyRunner` and `MotionTrackingOnPolicyRunner` automatically export ONNX models alongside PyTorch checkpoints:

- **Velocity runner**: Exports `policy.onnx` — a standard policy ONNX.
- **Tracking runner**: Exports TWO ONNX models:
  - `policy.onnx` — actor-only policy (same format as velocity).
  - `<dirname>.onnx` — bundles the actor policy AND reference motion data (joint_pos, joint_vel, body_pos_w, etc.) into a single ONNX model for deployment.

Training metrics are logged via **wandb** (when available) or locally. The runner also logs git metadata (commit hash, diff) to the run directory.

### Viewers

`scripts/play.py` supports two viewer backends: `native` (MuJoCo's built-in viewer, requires display) and `viser` (web-based, works headless). Auto-detection picks based on `DISPLAY`/`WAYLAND_DISPLAY` env vars.

## Key Constraints

- `--env.scene.num-envs` is the most critical training parameter — it sets parallel environment count and GPU memory scales directly with it. 4096 is a typical value but must be tuned per-GPU.
- MuJoCo uses EGL for headless rendering during training (`MUJOCO_GL=egl` is set automatically).
- Motion files for tracking must be pre-converted from CSV to NPZ format.
- Robot deployment requires a gamepad connected to the robot for mode switching (L2+R2 to enter debug mode, then the control program takes over).
