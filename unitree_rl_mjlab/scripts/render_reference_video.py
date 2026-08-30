"""Render a reference motion npz as a robot video (no policy, no RL env).

Plays back the reference (the GMR-mapped robot motion) kinematically: each
frame sets the robot qpos from the npz, does a forward pass, and renders with
an offscreen mujoco renderer. Requires a GL context, e.g.:

  xvfb-run -a python scripts/render_reference_video.py \
    --motion src/assets/motions/g1/0fdba7c6be1334e3b7a8d4c6dbf807ff.npz \
    --output ref_render.mp4 --width 640 --height 360
"""
import argparse
from pathlib import Path

import mujoco as mj
import mediapy as media
import numpy as np

G1_SCENE = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--motion", required=True, help="reference npz path")
  parser.add_argument("--output", default="ref_render.mp4")
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=360)
  parser.add_argument("--fps", type=float, default=None)
  parser.add_argument("--start", type=int, default=0, help="start frame")
  parser.add_argument("--end", type=int, default=None, help="end frame (exclusive)")
  args = parser.parse_args()

  d = np.load(args.motion)
  bp = d["body_pos_w"]
  bq = d["body_quat_w"]
  jp = d["joint_pos"]
  fps = args.fps or (float(d["fps"][0]) if "fps" in d else 50.0)

  start = args.start
  end = args.end or len(jp)
  print(f"frames: {end - start} (of {len(jp)}), fps={fps}")

  model = mj.MjModel.from_xml_path(str(G1_SCENE))
  data = mj.MjData(model)
  renderer = mj.Renderer(model, height=args.height, width=args.width)

  cam = mj.MjvCamera()
  cam.distance = 2.8
  cam.elevation = -20
  cam.azimuth = 90

  frames = []
  for i in range(start, end):
    data.qpos[:3] = bp[i, 0]
    data.qpos[3:7] = bq[i, 0]
    data.qpos[7:] = jp[i]
    mj.mj_forward(model, data)
    cam.lookat = bp[i, 0].copy()  # follow the robot
    cam.lookat[2] = 0.8
    renderer.update_scene(data, cam)
    frames.append(renderer.render().copy())
    if (i - start + 1) % 500 == 0 or i == end - 1:
      print(f"  rendered {i - start + 1}/{end - start}")

  media.write_video(args.output, frames, fps=fps)
  print(f"Done. Video saved to {args.output} ({len(frames)} frames @ {fps}fps)")


if __name__ == "__main__":
  main()
