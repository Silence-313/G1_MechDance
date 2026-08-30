#!/usr/bin/env python
"""Bias a tracking npz into a half-squat so the knees never hyperextend.

Motivation: the original reference motions keep the legs near-straight (knee
~0 rad, i.e. thigh/calf ~180 deg). On the real G1 that reads as knee
hyperextension, and the robot can only recentre by jittering its legs. We want
the robot to hold a half-squat, so the sagittal leg joints get a constant bias:

  * hip_pitch   (user motor 1)  -> more hip flexion   (negative offset)
  * knee        (user motor 4)  -> more knee flexion   (positive offset)
  * ankle_pitch (user motor 5)  -> more dorsiflexion   (negative offset)

applied to BOTH legs, then clamped to the joint limits.

CRITICAL (see memory npz-consistency): you must never edit joint_pos without
recomputing body FK. This script also LOWERS the root (pelvis) each frame so
the feet stay anchored to their ORIGINAL world height -- i.e. the dance's
footwork (steps/jumps/foot-lifts) is preserved exactly while the legs crouch
underneath. Then body_pos_w / body_quat_w / velocities are all recomputed from
the modified joints, so the reference stays physically self-consistent.

Leg joint order (0-based): 0 L_hip_pitch 1 L_hip_roll 2 L_hip_yaw 3 L_knee
4 L_ankle_pitch 5 L_ankle_roll, right leg 6..11 symmetric.

Usage:
  python deploy/robots/g1/squat_bias_npz.py <input.npz> <output.npz> \
      [--hip-offset -0.2] [--knee-offset 0.35] [--ankle-offset -0.15] \
      [--knee-floor 0.30]
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

G1_SCENE = (
    Path(__file__).resolve().parents[3]
    / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
)

# Sagittal "squat" joints: name -> (left_col, right_col, default_offset_rad).
SAGITTAL = {
    "hip_pitch": (0, 6, -0.20),
    "knee": (3, 9, +0.35),
    "ankle_pitch": (4, 10, -0.15),
}
# Joint limits (rad) from g1.xml, symmetric left/right.
LIMITS = {
    "hip_pitch": (-2.5307, 2.8798),
    "knee": (-0.087267, 2.8798),
    "ankle_pitch": (-0.87267, 0.5236),
}
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return q * np.array([1, -1, -1, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="original tracking npz")
    ap.add_argument("output", help="output squat-biased npz")
    ap.add_argument("--hip-offset", type=float, default=-0.20)
    ap.add_argument("--knee-offset", type=float, default=0.35)
    ap.add_argument("--ankle-offset", type=float, default=-0.15)
    ap.add_argument("--knee-floor", type=float, default=0.30,
                    help="hard knee minimum rad after offset (no straight leg)")
    ap.add_argument("--no-root-comp", action="store_true",
                    help="skip foot-anchored root lowering (feet may sink)")
    args = ap.parse_args()

    offsets = {
        "hip_pitch": args.hip_offset,
        "knee": args.knee_offset,
        "ankle_pitch": args.ankle_offset,
    }

    d = np.load(args.input)
    jp = d["joint_pos"].astype(np.float64)
    jp_orig = jp.copy()
    dt = 1.0 / float(d["fps"][0])
    root_pos = d["body_pos_w"][:, 0].astype(np.float64)  # pelvis world pos (N,3)
    root_quat = d["body_quat_w"][:, 0].astype(np.float64)  # pelvis quat (N,4)
    bp_orig = d["body_pos_w"].astype(np.float64)  # (N, nbody, 3)

    model = mj.MjModel.from_xml_path(str(G1_SCENE))
    data = mj.MjData(model)
    n_bodies = model.nbody - 1  # excludes world
    foot_id = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
    foot_col = [f - 1 for f in foot_id]  # index into body_pos_w (world excluded)

    # ---- 1. Modify sagittal joints: offset + clip + knee floor ----
    print(f"input: {len(jp)} frames, fps={1.0/dt:.1f}, joint_pos {jp.shape}")
    for name, (lc, rc, _) in SAGITTAL.items():
        lo, hi = LIMITS[name]
        off = offsets[name]
        for c in (lc, rc):
            col = jp_orig[:, c].copy()
            new = np.clip(col + off, lo, hi)
            if name == "knee":
                new = np.maximum(new, args.knee_floor)
            jp[:, c] = new
            print(f"  {name:11s} col{c:2d} off{off:+7.3f}  "
                  f"min {col.min():+8.3f}->{new.min():+8.3f}  "
                  f"med {np.median(col):+8.3f}->{np.median(new):+8.3f}  "
                  f"max {col.max():+8.3f}->{new.max():+8.3f}")

    # Original foot min-z per frame (read straight from the source body_pos_w).
    foot_z_orig = bp_orig[:, foot_col, 2].min(axis=1)

    # ---- 2. Foot-anchored root lowering ----
    if not args.no_root_comp:
        for i in range(len(jp)):
            q = np.concatenate([root_pos[i], root_quat[i], jp[i]])
            data.qpos[:] = q
            mj.mj_forward(model, data)
            foot_z_tmp = data.xpos[foot_id, 2].min()
            root_pos[i, 2] -= foot_z_tmp - foot_z_orig[i]
        dz = root_pos[:, 2] - d["body_pos_w"][:, 0, 2].astype(np.float64)
        print(f"  root z shift: min {dz.min():+.3f} med {np.median(dz):+.3f} "
              f"max {dz.max():+.3f} m")

    # ---- 3. Final FK -> body_pos_w / body_quat_w ----
    body_pos_w = np.zeros((len(jp), n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((len(jp), n_bodies, 4), dtype=np.float32)
    for i in range(len(jp)):
        q = np.concatenate([root_pos[i], root_quat[i], jp[i]])
        data.qpos[:] = q
        mj.mj_forward(model, data)
        body_pos_w[i] = data.xpos[1:]
        body_quat_w[i] = data.xquat[1:]

    if not args.no_root_comp:
        err = np.abs(body_pos_w[:, foot_col, 2].min(axis=1) - foot_z_orig).max()
        print(f"  foot-anchor max residual: {err:.4f} m")

    # ---- 4. Velocities via finite differences (same scheme as add_standing) ----
    jp_t = torch.from_numpy(jp)
    bp_t = torch.from_numpy(body_pos_w.astype(np.float64))
    joint_vel = torch.gradient(jp_t, spacing=dt, dim=0)[0].numpy()
    body_lin_vel_w = torch.gradient(bp_t, spacing=dt, dim=0)[0].numpy()

    B, J, _ = body_pos_w.shape
    body_ang_vel_w = np.zeros((B, J, 3), dtype=np.float32)
    for i in range(B):
        q_prev = body_quat_w[max(i - 1, 0)]
        q_next = body_quat_w[min(i + 1, B - 1)]
        q_rel = np.stack([quat_mul(q_next[j], quat_conj(q_prev[j])) for j in range(J)])
        axis_angle = 2.0 * np.arctan2(np.linalg.norm(q_rel[:, 1:], axis=1), q_rel[:, 0])
        axis = q_rel[:, 1:] / np.maximum(np.linalg.norm(q_rel[:, 1:], axis=1, keepdims=True), 1e-8)
        body_ang_vel_w[i] = (axis * axis_angle[:, None]) / (2.0 * dt)

    jv_max = float(np.abs(d["joint_vel"]).max()) * 1.5
    blv_max = float(np.abs(d["body_lin_vel_w"]).max()) * 1.5
    bav_max = float(np.abs(d["body_ang_vel_w"]).max()) * 1.5
    joint_vel = np.clip(joint_vel, -jv_max, jv_max)
    body_lin_vel_w = np.clip(body_lin_vel_w, -blv_max, blv_max)
    body_ang_vel_w = np.clip(body_ang_vel_w, -bav_max, bav_max)

    out = {
        "fps": d["fps"],
        "joint_pos": jp.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w,
    }
    np.savez(args.output, **out)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
