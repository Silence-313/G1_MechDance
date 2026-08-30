# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **MechDance** — a humanoid robot motion transfer platform. It contains three interconnected projects that form a pipeline from video to physical robot:

```
Video of human  ──▶  [GVHMR]  ──▶  SMPL-X  ──▶  [GMR]  ──▶  robot joints  ──▶  [unitree_rl_mjlab]  ──▶  physical robot
                     (pose est.)        (human motion)     (retargeting)      (robot motion)      (RL policy)
```

- **GVHMR** — recovers 3D human motion in world coordinates from monocular video (SIGGRAPH Asia 2024)
- **GMR** — retargets human motion (SMPL-X/BVH/FBX) to humanoid robot joint positions via real-time inverse kinematics (MIT licensed)
- **unitree_rl_mjlab** — trains RL policies for locomotion and motion tracking on Unitree robots, with sim-to-real ONNX deployment

Each subproject has its own `CLAUDE.md` with detailed architecture, commands, and conventions. The three projects are otherwise independent — each has its own conda environment (`gvhmr`, `gmr`, `unitree_rl_mjlab`), its own `pip install -e .` editable-install (Python 3.10 for GMR/GVHMR, 3.11 for unitree_rl_mjlab), its own dependencies, and no cross-imports.

## Cross-Project Data Flow

The projects chain together through **file formats, not imports** — there is no Python-level dependency between any two of them. The handoff points are:

```
monocular video
  → [GVHMR]             → hmr4d_results.pt  (SMPL-X predictions, world coordinates)
  → [GMR]               → .pkl (root_pos, root_rot, dof_pos)
  → [pkl→csv→npz]       → .npz motion file (joint_pos, joint_vel, body_pos_w, ...), 50 fps
  → [unitree_rl_mjlab]  → RL policy (PPO) → ONNX → robot
```

- **GVHMR → GMR**: GVHMR's `demo.py` writes `outputs/demo/<video_stem>/hmr4d_results.pt`. GMR's `scripts/gvhmr_to_robot.py --gvhmr_pred_file=<that .pt> --robot unitree_g1 --save_path <out.pkl>` consumes it directly — no intermediate SMPL-X file is saved. (`smplx_to_robot.py` / `bvh_to_robot.py` are the equivalents for AMASS `.npz` / BVH inputs.)
- **GMR output**: a `.pkl` dict with keys `fps`, `root_pos` (N×3), `root_rot` (N×4 xyzw), `dof_pos` (N×29) — the base translation/rotation and joint positions, per frame.
- **GMR → unitree_rl_mjlab**: the RL tracking task (`--motion_file=...npz`) cannot consume `.pkl` directly. Convert `.pkl → .csv` with GMR's `scripts/batch_gmr_pkl_to_csv.py --folder <dir>` (36 columns: root_pos 3 + root_rot 4 + dof_pos 29, downsampled to 30 fps), then `.csv → .npz` with unitree_rl_mjlab's `scripts/csv_to_npz.py --input-fps 30 --output-fps 50 --robot g1`.
- The working reference motion for `Unitree-G1-Tracking-No-State-Estimation` is `unitree_rl_mjlab/src/assets/motions/g1/test_g1_motion.npz` (587 frames @ 50 fps), produced through this exact chain.

**NPZ consistency (critical):** a tracking `.npz` is consumed both at train time (rewards) and deploy time (`MotionLoader_` reads `body_pos_w`/`body_quat_w`/`joint_pos`/`joint_vel`). Never hand-edit `joint_pos` (e.g. clamp knee angles) without recomputing the `body_*` fields via forward kinematics — otherwise the reference becomes self-contradictory and the policy converges to high-frequency jitter that trips the robot's BMS ("弹电池"). Use `deploy/robots/g1/regenerate_consistent_npz.py` to recompute FK after any joint edit.

### One-shot end-to-end script

`unitree_rl_mjlab/run_video_to_rl.sh` runs the whole chain (video → GVHMR → GMR → CSV → NPZ → RL training):

```bash
./unitree_rl_mjlab/run_video_to_rl.sh --video /path/to/dance.mp4 --name my_dance --num-envs 1024
```

Flags: `--robot g1|g1_23dof`, `--input-fps <N>` (default 30) / `--output-fps <N>` (default 50), `--no-gvhmr` / `--no-gmr` / `--no-train` (reuse existing intermediate artifacts), `--keep-tmp`.

**Caveat**: the script's default `GVHMR_DIR`/`GMR_DIR` point to `/home/silence/GVHMR` and `/home/silence/GMR`, which do **not** exist on this machine (projects live under `/home/silence/MechDance/`). Override them:

```bash
GVHMR_DIR=/home/silence/MechDance/GVHMR GMR_DIR=/home/silence/MechDance/GMR \
  ./unitree_rl_mjlab/run_video_to_rl.sh --video /path/to/dance.mp4
```

The script activates the `gvhmr`, `gmr`, and `unitree_rl_mjlab` conda envs in sequence (one subprocess per stage), so all three environments must exist locally.

## Development Environment

- Only `unitree_rl_mjlab/` is a git repository in this working copy; `GMR/` and `GVHMR/` are plain source trees (no `.git`).
- Training, GPU/EGL rendering, and robot deployment run on a separate **offline server**; this machine is for editing code and preparing data. Changes are made here and manually uploaded by the user — don't attempt GPU training locally.
- The server↔robot connection/deployment procedure (SSH, IPs, netplan, `scp` upload, DDS run) is documented in `11.md` (repo root). The root script `fix_robot_network.sh` automates the netplan steps from that doc (run `sudo ./fix_robot_network.sh` on the server).
- There are no automated tests (or CI) in any of the three projects — correctness is verified by running the pipeline end-to-end and inspecting outputs (GVHMR `.pt`, GMR `.pkl`/`.csv`, RL tfevents/wandb logs).

## Key reference files & docs

Beyond the three subproject CLAs, these capture non-obvious workflow and state knowledge:

- `11.md` (repo root) — step-by-step guide for connecting this machine to the offline server (WiFi `10.101.x.x`) and the server to the robot (Ethernet `192.168.123.99`), then uploading and running the control binary.
- `unitree_rl_mjlab/deploy/robots/g1/DEPLOYMENT.md` — canonical guide for deploying a trained motion to the G1: the "3-file bundle" (`exported/policy.onnx` + `params/deploy.yaml` + `params/<name>.npz`) + 3-spot `config.yaml` FSM registration + `scp` upload + `./g1_ctrl --network=enp6s0` + gamepad trigger (L2+R2 hardware debug mode, then RB+button to play). Note: `192.168.123.99` in 11.md is the *server's own* netplan-set `enp6s0` IP — the robot itself has been observed at `192.168.123.161`; confirm with `ip a`/`arp` before assuming.
- `unitree_rl_mjlab/deploy/robots/g1/DEPLOYMENT_STANDALONE.md` — **standalone (server-less) mode** (2026-08-20, verified working): compile & run `g1_ctrl` on the robot's onboard **PC2** (dev computer, `unitree@192.168.123.164`, pwd `123`, aarch64 Ubuntu 20.04, network iface `eth0`) so the robot dances with no server/cable/WiFi. Covers the 4 build pitfalls (arm64 libfmt deb, new-SDK2 `dds_wrapper` headers, new-SDK2 aarch64 `libunitree_sdk2.a`, scp-broken onnxruntime symlinks), systemd auto-start (`g1_ctrl.service`, `Restart=always`), and gamepad ops. PC2 only accepts SSH from the server subnet — all files must go local → server → PC2.
- **Frozen dances** (2026-08-17): `957d1c6f` and `17fe0adf` are final and deployed — do not modify them without explicit user confirmation. Current `config.yaml` button map (2026-08-20): RB+A=`957d1c6f` (frozen), RB+B=`17fe0adf` (frozen), RB+X=`93ccdf54` (raw), RB+Y=`c2197703` (raw). This same map is mirrored on the onboard PC2.
- `unitree_rl_mjlab/RL_CURRENT_STATUS.md` — current G1 motion-tracking training state (motion file, curriculum, thresholds). Companion docs in the same dir: `RL_PARAMETER_AUDIT.md`, `RL_FIX_CHANGELOG.md`, `RL_FIX_VALIDATION_REPORT.md`.
- `GMR/TEST_MOTIONS.md` — motions known to retarget poorly; `GMR/DOC.md` — annotation of the IK-config JSON format (`ik_match_table1`).
- Root-level `.npz` files (`0fdba7c6…npz`, `…_standing.npz`) are reference motions in current use; `deploy.yaml`, `111/`, and `model_69999.pt` are deployment/policy artifacts, not source.

## Subproject CLAs

- [GMR/CLAUDE.md](GMR/CLAUDE.md) — motion retargeting (IK), human-to-robot body mapping, BVH/SMPL-X/FBX input formats
- [GVHMR/CLAUDE.md](GVHMR/CLAUDE.md) — video pose estimation, Hydra config system, PyTorch Lightning training
- [unitree_rl_mjlab/CLAUDE.md](unitree_rl_mjlab/CLAUDE.md) — RL training (PPO via mjlab), task registration, C++ deployment
