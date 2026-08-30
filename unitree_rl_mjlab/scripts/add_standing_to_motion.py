"""Prepend/append standing segments to a tracking reference npz.

The reference starts from a deep squat (frame 0), which is a hard initial pose
for the RSI "start" mode. This script rebuilds the motion as:

    standing (hold) -> transition into dance-start (squat) -> dance -> transition back to standing -> standing (hold)

so frame 0 is a standing pose, the robot crouches into the squat, dances, then
stands back up. Output is a new npz with the same schema as the input
(joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w).

Usage:
  python scripts/add_standing_to_motion.py \
    --input src/assets/motions/g1/<motion>.npz \
    --output src/assets/motions/g1/<motion>_standing.npz \
    --stand-before 100 --transition-in 100 --transition-out 100 --stand-after 100
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

G1_SCENE = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

# Standing home pose (HOME_KEYFRAME in g1_constants.py), in csv_to_npz joint order.
STANDING_DOF = np.array([
  # left  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  # right
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  # waist yaw, roll, pitch
  0.0, 0.0, 0.0,
  # left shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw
  0, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
  # right
  0, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
], dtype=np.float32)
# Arms-down standing rest pose for the start/end. User-specified: shoulder_pitch
# 0.3 (avoids the 0.70 backward hyperextension), shoulder_roll -0.3 (slight
# open), elbow 1.2. Note G1's shoulder_roll is mirrored (left/right opposite),
# so both set to -0.3 may look asymmetric — mirror the right if needed.
STANDING_DOF_DOWN = np.array([
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  0.0, 0.0, 0.0,
  0, 0.3, 0.0, 1.2, 0.0, 0.0, 0.0,
  0, -0.3, 0.0, 1.2, 0.0, 0.0, 0.0,
], dtype=np.float32)
STANDING_ROOT_Z = 0.8


def yaw_only_quat(q_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = q_wxyz
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
  q0 = q0 / np.linalg.norm(q0)
  q1 = q1 / np.linalg.norm(q1)
  d = float(np.dot(q0, q1))
  if d < 0.0:
    q1 = -q1
    d = -d
  if d > 0.9995:
    return q0 + t * (q1 - q0)
  theta = np.arccos(np.clip(d, -1.0, 1.0))
  return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def build_qpos(root_pos, root_quat, dof):
  return np.concatenate([root_pos, root_quat, dof])


def ground_pose(q, model, data) -> np.ndarray:
  """Lower the root so the lowest body (feet) touches z=0."""
  data.qpos[:] = q
  mj.mj_forward(model, data)
  foot_z = data.xpos[1:, 2].min()
  q = q.copy()
  q[2] -= foot_z
  data.qpos[:] = q
  mj.mj_forward(model, data)
  return q


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument(
    "--cut-before",
    type=int,
    default=0,
    help="Cut the first N frames of the input dance (e.g. to remove the "
    "initial deep-squat start). The standing intro then transitions into "
    "whatever remains.",
  )
  parser.add_argument(
    "--end-at-highest-arm",
    action="store_true",
    help="Cut the dance at the frame where the arms are at their highest point, "
    "and use that pose as the dance's ending (then transition to standing).",
  )
  parser.add_argument(
    "--end-frame",
    type=int,
    default=None,
    help="Cut the dance at this ORIGINAL frame (inclusive) and use it as the "
    "ending pose (then transition to standing). E.g. 3007 = both-hands-up peak.",
  )
  parser.add_argument("--stand-before", type=int, default=100)
  parser.add_argument("--transition-in", type=int, default=100)
  parser.add_argument("--transition-out", type=int, default=100)
  parser.add_argument("--stand-after", type=int, default=100)
  parser.add_argument(
    "--end-hold",
    type=int,
    default=0,
    help="Hold the dance's ending pose for N frames before transitioning back "
    "to the arms-down standing pose (0 = go straight to the transition).",
  )
  args = parser.parse_args()

  d = np.load(args.input)
  bp = d["body_pos_w"]
  bq = d["body_quat_w"]
  jp = d["joint_pos"]
  dt = 1.0 / float(d["fps"][0]) if "fps" in d else 0.02
  if args.cut_before > 0:
    bp = bp[args.cut_before:]
    bq = bq[args.cut_before:]
    jp = jp[args.cut_before:]
    print(f"cut first {args.cut_before} frames (squat start) -> {len(jp)} frames left")
  if args.end_at_highest_arm:
    model0 = mj.MjModel.from_xml_path(str(G1_SCENE))
    lw = mj.mj_name2id(model0, mj.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link") - 1
    rw = mj.mj_name2id(model0, mj.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link") - 1
    hi = np.maximum(bp[:, lw, 2], bp[:, rw, 2])
    peak = int(np.argmax(hi))
    bp = bp[: peak + 1]
    bq = bq[: peak + 1]
    jp = jp[: peak + 1]
    print(f"end at highest-arm frame {peak} (wrist z={hi[peak]:.3f})")
  if args.end_frame is not None:
    cut_to = args.end_frame - args.cut_before + 1
    bp = bp[:cut_to]
    bq = bq[:cut_to]
    jp = jp[:cut_to]
    print(f"cut dance to original frame {args.end_frame} (cut-frame {cut_to - 1})")
  print(f"input: {len(jp)} frames, fps={1.0/dt:.1f}")

  # Reconstruct the full qpos trajectory.
  n_frames = len(jp)
  qpos_all = np.stack(
    [build_qpos(bp[f, 0], bq[f, 0], jp[f]) for f in range(n_frames)]
  )
  dof_all = jp

  # Standing pose anchored at the dance start / end (same xy, upright, yaw aligned).
  start_xy = bp[0, 0, :2]
  end_xy = bp[-1, 0, :2]
  start_yaw_q = yaw_only_quat(bq[0, 0])
  end_yaw_q = yaw_only_quat(bq[-1, 0])

  def standing_qpos(xy, yaw_q, dof=STANDING_DOF):
    return build_qpos(
      np.array([xy[0], xy[1], STANDING_ROOT_Z], dtype=np.float32),
      yaw_q.astype(np.float32),
      dof,
    )

  # Both the starting and ending standing poses use the arms-down rest pose
  # (no raised hands).
  q_start_stand = standing_qpos(start_xy, start_yaw_q, STANDING_DOF_DOWN)
  q_end_stand = standing_qpos(end_xy, end_yaw_q, STANDING_DOF_DOWN)

  model = mj.MjModel.from_xml_path(str(G1_SCENE))
  data = mj.MjData(model)
  q_start_stand = ground_pose(q_start_stand, model, data)
  q_end_stand = ground_pose(q_end_stand, model, data)

  # Assemble: stand -> transition-in -> dance -> transition-out -> stand.
  def lerp_seg(a, b, n):
    return np.stack([a + (b - a) * t for t in np.linspace(0, 1, n)])

  def slerp_seg(q0, q1, n):
    return np.stack([slerp(q0, q1, t).astype(np.float32) for t in np.linspace(0, 1, n)])

  def interp_qpos(qa, qb, n):
    if n <= 0:
      # Hard cut: no transition frames. The agent must learn the connection
      # (e.g. standing -> squat) itself during RL.
      return np.zeros((0, len(qa)), dtype=np.float32)
    seg = np.zeros((n, len(qa)), dtype=np.float32)
    seg[:, :3] = lerp_seg(qa[:3], qb[:3], n)
    seg[:, 3:7] = slerp_seg(qa[3:7], qb[3:7], n)
    seg[:, 7:] = lerp_seg(qa[7:], qb[7:], n)
    return seg

  pieces = []
  pieces.append(np.tile(q_start_stand, (args.stand_before, 1)))
  pieces.append(interp_qpos(q_start_stand, qpos_all[0], args.transition_in))
  pieces.append(qpos_all)
  # Hold the dance's ending pose, then transition back to the standing pose.
  pieces.append(np.tile(qpos_all[-1], (args.end_hold, 1)))
  pieces.append(interp_qpos(qpos_all[-1], q_end_stand, args.transition_out))
  pieces.append(np.tile(q_end_stand, (args.stand_after, 1)))
  qpos_new = np.concatenate(pieces, axis=0)
  print(f"output: {len(qpos_new)} frames")

  # FK all new frames -> body_pos_w / body_quat_w (model body 1..30).
  n_bodies = model.nbody - 1  # exclude world
  body_pos_w = np.zeros((len(qpos_new), n_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((len(qpos_new), n_bodies, 4), dtype=np.float32)
  for i, q in enumerate(qpos_new):
    data.qpos[:] = q
    mj.mj_forward(model, data)
    body_pos_w[i] = data.xpos[1:]
    body_quat_w[i] = data.xquat[1:]

  # Velocities via finite differences.
  jp_t = torch.from_numpy(qpos_new[:, 7:])
  bp_t = torch.from_numpy(body_pos_w)
  joint_vel = torch.gradient(jp_t, spacing=dt, dim=0)[0].numpy()
  body_lin_vel_w = torch.gradient(bp_t, spacing=dt, dim=0)[0].numpy()

  # Angular velocity from quaternion relative rotation (same scheme as csv_to_npz).
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

  B, J, _ = body_pos_w.shape
  body_ang_vel_w = np.zeros((B, J, 3), dtype=np.float32)
  for i in range(B):
    q_prev = body_quat_w[max(i - 1, 0)]
    q_next = body_quat_w[min(i + 1, B - 1)]
    q_rel = np.stack([quat_mul(q_next[j], quat_conj(q_prev[j])) for j in range(J)])
    axis_angle = 2.0 * np.arctan2(
      np.linalg.norm(q_rel[:, 1:], axis=1), q_rel[:, 0]
    )
    axis = q_rel[:, 1:] / np.maximum(np.linalg.norm(q_rel[:, 1:], axis=1, keepdims=True), 1e-8)
    body_ang_vel_w[i] = (axis * axis_angle[:, None]) / (2.0 * dt)

  # Clamp boundary velocity spikes (from hard cuts between segments) to the
  # original dance's real velocity range. The agent must still move fast to
  # connect the poses, but the reference velocity targets stay physically
  # achievable instead of one-frame infinities.
  jv_max = float(np.abs(d["joint_vel"]).max()) * 1.5
  blv_max = float(np.abs(d["body_lin_vel_w"]).max()) * 1.5
  bav_max = float(np.abs(d["body_ang_vel_w"]).max()) * 1.5
  joint_vel = np.clip(joint_vel, -jv_max, jv_max)
  body_lin_vel_w = np.clip(body_lin_vel_w, -blv_max, blv_max)
  body_ang_vel_w = np.clip(body_ang_vel_w, -bav_max, bav_max)

  out = {
    "fps": np.array([1.0 / dt]),
    "joint_pos": qpos_new[:, 7:].astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
  }
  np.savez(args.output, **out)
  print(f"saved: {args.output} ({len(qpos_new)} frames)")


if __name__ == "__main__":
  main()
