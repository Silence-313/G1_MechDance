#!/usr/bin/env python
"""Regenerate a knee-clamped reference npz with FK-consistent body data.

fix_knee_npz.py only clamps joint_pos (knees to >=knee_min) but leaves
body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w computed from the
ORIGINAL straight-knee joints untouched. That makes the reference internally
inconsistent: the joint angles say "bend knees 30 deg" while the body-link
positions say "legs straight". A policy that must simultaneously satisfy the
body-tracking rewards (from body_pos_w) and the knee reward (from joint_pos)
cannot do both -> it converges to a chattering compromise -> real-robot motor
overcurrent / BMS trip.

This script clamps the knees and RECOMPUTES all body FK (positions, quats,
linear/angular velocities) from the clamped joints, so the reference is
physically self-consistent.

Usage:
  python deploy/robots/g1/regenerate_consistent_npz.py \
    <input.npz> <output.npz> [--knee-min 0.52]

FK machinery mirrors scripts/add_standing_to_motion.py (verified exact vs
mujoco FK).
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
KNEE_COLS = (3, 9)


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
    ap.add_argument("input", help="original (or knee_bend) tracking npz")
    ap.add_argument("output", help="output consistent npz")
    ap.add_argument("--knee-min", type=float, default=0.52, help="knee floor rad")
    args = ap.parse_args()

    d = np.load(args.input)
    jp = d["joint_pos"].astype(np.float64)
    jp_orig = jp.copy()
    for c in KNEE_COLS:
        jp[:, c] = np.maximum(jp[:, c], args.knee_min)
    changed = int((np.abs(jp - jp_orig) > 1e-6).any(axis=1).sum())
    print(f"knee clamp {args.knee_min}: {changed} frames changed")
    for c in KNEE_COLS:
        print(
            f"  col{c}: min {jp_orig[:, c].min():+.3f}->{jp[:, c].min():+.3f}, "
            f"median {np.median(jp_orig[:, c]):+.3f}->{np.median(jp[:, c]):+.3f}"
        )

    dt = 1.0 / float(d["fps"][0])
    root_pos = d["body_pos_w"][:, 0].astype(np.float64)
    root_quat = d["body_quat_w"][:, 0].astype(np.float64)
    qpos = np.concatenate([root_pos, root_quat, jp], axis=1)
    print(f"frames {len(qpos)}, dt {dt:.5f}s")

    model = mj.MjModel.from_xml_path(str(G1_SCENE))
    data = mj.MjData(model)
    n_bodies = model.nbody - 1
    body_pos_w = np.zeros((len(qpos), n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((len(qpos), n_bodies, 4), dtype=np.float32)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        body_pos_w[i] = data.xpos[1:]
        body_quat_w[i] = data.xquat[1:]

    jp_t = torch.from_numpy(qpos[:, 7:])
    bp_t = torch.from_numpy(body_pos_w)
    joint_vel = torch.gradient(jp_t, spacing=dt, dim=0)[0].numpy()
    body_lin_vel_w = torch.gradient(bp_t, spacing=dt, dim=0)[0].numpy()

    B, J, _ = body_pos_w.shape
    body_ang_vel_w = np.zeros((B, J, 3), dtype=np.float32)
    for i in range(B):
        q_prev = body_quat_w[max(i - 1, 0)]
        q_next = body_quat_w[min(i + 1, B - 1)]
        q_rel = np.stack([quat_mul(q_next[j], quat_conj(q_prev[j])) for j in range(J)])
        axis_angle = 2.0 * np.arctan2(np.linalg.norm(q_rel[:, 1:], axis=1), q_rel[:, 0])
        axis = q_rel[:, 1:] / np.maximum(
            np.linalg.norm(q_rel[:, 1:], axis=1, keepdims=True), 1e-8
        )
        body_ang_vel_w[i] = (axis * axis_angle[:, None]) / (2.0 * dt)

    # Clamp boundary velocity spikes (same as add_standing).
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
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    np.savez(args.output, **out)
    print(f"saved {args.output}")

    # Sanity: knee mismatch vs body positions now consistent.
    for c in KNEE_COLS:
        print(f"  final col{c}: min {jp[:, c].min():+.3f} median {np.median(jp[:, c]):+.3f}")


if __name__ == "__main__":
    main()
