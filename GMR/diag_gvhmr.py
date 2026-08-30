"""
诊断脚本：在服务器上运行，输出 GVHMR -> SMPL-X 全链路的坐标系和高度信息。
用法（在服务器 GMR 目录下）:
    python diag_gvhmr.py <hmr4d_results.pt路径>
"""
import sys
import torch
import numpy as np
import smplx
from scipy.spatial.transform import Rotation as R

gvhmr_file = sys.argv[1]
print("=" * 60)
print("诊断: GVHMR -> SMPL-X 坐标系与高度")
print("=" * 60)

data = torch.load(gvhmr_file, map_location="cpu", weights_only=False)
sp = data["smpl_params_global"]
nf = sp["body_pose"].shape[0]
print(f"帧数: {nf}")
print(f"betas[0] (GVHMR, 10维): {sp['betas'][0].numpy()}")
beta0 = sp["betas"][0, 0].item()
human_height = 1.66 + 0.1 * beta0
print(f"估算人体身高: {human_height:.3f} m")

bm = smplx.create("assets/body_models", "smplx", gender="neutral", use_pca=False)
need_betas = bm.num_betas
print(f"smplx 期望 betas 维度: {need_betas}")
print(f"smplx shapedirs: {bm.shapedirs.shape}")

# 将 GVHMR 的 10 维 betas pad 到 smplx 需要的维度
def pad_betas(b):
    if len(b.shape) == 2:
        b = b[0]
    b = np.asarray(b)
    if b.shape[0] < need_betas:
        return np.pad(b, (0, need_betas - b.shape[0]))
    return b[:need_betas]

beta = pad_betas(sp["betas"].numpy())
print(f"pad 后 betas 维度: {beta.shape}")

def bbox_of(go, tl, label):
    out = bm(
        betas=torch.tensor(beta).float().view(1, -1).repeat(nf, 1),
        global_orient=torch.tensor(go).float(),
        body_pose=torch.tensor(sp["body_pose"].numpy()).float(),
        transl=torch.tensor(tl).float(),
        left_hand_pose=torch.zeros(nf, 45), right_hand_pose=torch.zeros(nf, 45),
        jaw_pose=torch.zeros(nf, 3), leye_pose=torch.zeros(nf, 3), reye_pose=torch.zeros(nf, 3),
        expression=torch.zeros(nf, bm.num_expression_coeffs),
        return_full_pose=True,
    )
    v = out.vertices[0].detach().numpy()
    j = out.joints[0].detach().numpy()
    print(f"\n[{label}] 第0帧:")
    for ax, name in [(0, "X"), (1, "Y"), (2, "Z")]:
        print(f"  顶点 {name}: [{v[:, ax].min():.3f}, {v[:, ax].max():.3f}] 范围 {v[:, ax].max()-v[:, ax].min():.3f}")
    print(f"  pelvis: {np.round(j[0], 3)}")
    print(f"  head  : {np.round(j[15], 3)}")
    print(f"  脚底Z: {min(j[10, 2], j[11, 2]):.3f}  头顶Z: {j[15, 2]:.3f}")
    return j

go = sp["global_orient"].numpy()
tl = sp["transl"].numpy()

# 1. 原始
bbox_of(go, tl, "原始(不旋转)")

# 2. 绕Z90
rc = R.from_rotvec([0, 0, np.pi / 2])
go_c = np.array([(rc * R.from_rotvec(g)).as_rotvec() for g in go])
tl_c = np.array([rc.apply(t) for t in tl])
bbox_of(go_c, tl_c, "绕Z90° (orient+transl)")

print("\n" + "=" * 60)
print("关键判断:")
print("如果[原始]身体沿X轴站立(顶点X范围~1.7m, Y/Z小) => GVHMR是X-up, 需要旋转")
print("如果[绕Z90]身体沿Y轴站立(顶点Y范围~1.7m) => 旋转正确, 身体直立")
print("  此时脚底Y应接近骨盆下方, 检查脚底Y值是否在地面附近")
