#!/usr/bin/env python
"""把动作 npz 的膝盖列 clamp 到微屈下限，避免机器人膝盖伸直锁死。

G1 腿部关节顺序（0-based，见 src/assets/robots/unitree_g1/xmls/g1.xml）：
  0 left_hip_pitch   1 left_hip_roll    2 left_hip_yaw
  3 left_knee        4 left_ankle_pitch  5 left_ankle_roll
  6 right_hip_pitch  7 right_hip_roll    8 right_hip_yaw
  9 right_knee      10 right_ankle_pitch 11 right_ankle_roll

膝盖限位 [-0.087, 2.88] rad：0=伸直，正值=屈膝。
"""
import argparse
import numpy as np

KNEE_COLS = (3, 9)  # left_knee, right_knee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入动作 npz")
    ap.add_argument("output", help="输出动作 npz")
    ap.add_argument("--knee-min", type=float, default=0.52,
                    help="膝盖微屈下限(rad)，默认 0.52≈30°")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="膝盖整体偏移(rad)，正=整体更屈，默认 0 纯 clamp")
    ap.add_argument("--cols", type=int, nargs="+", default=list(KNEE_COLS),
                    help="膝盖列索引，默认 3 9")
    args = ap.parse_args()

    d = np.load(args.input)
    jp = d["joint_pos"].astype(np.float64)
    jv = d["joint_vel"].astype(np.float64)

    for c in args.cols:
        col = jp[:, c].copy()
        newcol = np.maximum(col + args.offset, args.knee_min)
        jp[:, c] = newcol
        print(f"col {c}: 偏移+{args.offset} clamp>= {args.knee_min}, "
              f"min {col.min():+.3f}->{newcol.min():+.3f}, "
              f"中位 {np.median(col):+.3f}->{np.median(newcol):+.3f}, "
              f"max {col.max():+.3f}->{newcol.max():+.3f}")

    out = {k: d[k] for k in d.files}
    out["joint_pos"] = jp
    out["joint_vel"] = jv
    np.savez(args.output, **out)
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
