#!/usr/bin/env python
"""Print sagittal leg-joint stats for each original motion npz.

Leg joint order (0-based): 0 L_hip_pitch 1 L_hip_roll 2 L_hip_yaw
3 L_knee 4 L_ankle_pitch 5 L_ankle_roll, right leg 6..11 symmetric.
Sagittal (the "squat" joints, user motors 1/4/5): hip_pitch, knee, ankle_pitch.
"""
import numpy as np

SAGITTAL = {
    "L_hip_pitch": 0, "L_knee": 3, "L_ankle_pitch": 4,
    "R_hip_pitch": 6, "R_knee": 9, "R_ankle_pitch": 10,
}

FILES = [
    "/home/silence/MechDance/unitree_rl_mjlab/src/assets/motions/g1/957d1c6fe4fcc0812b69b301cc6da20c.npz",
    "/home/silence/MechDance/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/17fe0adfa5f2768305e3a7b2bfe92553.npz",
    "/home/silence/MechDance/93ccdf54051dfc119a6dc94b617ddc39.npz",
    "/home/silence/MechDance/c21977039c8dc2a52e5d5d9f4d5a2382.npz",
    "/home/silence/MechDance/unitree_rl_mjlab/src/assets/motions/g1/test_g1_motion.npz",
]


def main():
    for f in FILES:
        d = np.load(f)
        jp = d["joint_pos"]
        print(f"\n=== {f.split('/')[-1]}  frames={len(jp)}  shape={jp.shape}")
        for name, c in SAGITTAL.items():
            col = jp[:, c]
            print(f"  {name:14s} col{c:2d}  min {col.min():+8.4f}  med {np.median(col):+8.4f}  max {col.max():+8.4f}  (deg min {np.degrees(col.min()):+7.1f} med {np.degrees(np.median(col)):+7.1f})")


if __name__ == "__main__":
    main()
