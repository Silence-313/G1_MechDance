"""
诊断 GMR 输出的 pkl：用 MuJoCo 检查机器人脚是否贴地。
用法（服务器 GMR 目录）:
    python diag_pkl.py <动作.pkl>
"""
import sys
import pickle
import numpy as np
import mujoco

pkl_file = sys.argv[1]
with open(pkl_file, "rb") as f:
    d = pickle.load(f)

model = mujoco.MjModel.from_xml_path("assets/unitree_g1/g1_mocap_29dof.xml")
data = mujoco.MjData(model)

rp = d["root_pos"]
rr = d["root_rot"]  # xyzw 存储
dof = d["dof_pos"]

lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_toe_link")
rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_toe_link")
pel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

print(f"帧数: {len(rp)}, root_pos Z: min={rp[:,2].min():.3f} max={rp[:,2].max():.3f} mean={rp[:,2].mean():.3f}")
print()

for i in [0, 50, 100, 200, 414]:
    data.qpos[:3] = rp[i]
    data.qpos[3:7] = rr[i][[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = dof[i]
    mujoco.mj_forward(model, data)
    lf = data.xpos[lf_id]
    rf = data.xpos[rf_id]
    pel = data.xpos[pel_id]
    print("frame %d: 骨盆Z=%.3f  左脚Z=%.3f  右脚Z=%.3f" % (i, pel[2], lf[2], rf[2]))

print()
print("判断: 脚Z 应接近 0 (贴地). 若脚Z 明显 >0 (如 >0.05), 则脚离地=悬空")
print("     骨盆Z 应接近 0.8m (G1站立). 若骨盆Z>0.86, 超出腿长支撑")
