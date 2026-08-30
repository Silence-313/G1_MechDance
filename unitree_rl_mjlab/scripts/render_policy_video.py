"""Batch-render a trained tracking policy to a video (no interactive viewer).

Renders ``video_length`` policy steps on the reference motion, writes an mp4,
and exits cleanly (no viewer, no Ctrl-C needed). Requires a GL context for the
offscreen renderer, e.g.:

  xvfb-run -a python scripts/render_policy_video.py Unitree-G1-Tracking-No-State-Estimation \
    --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_<iter>.pt \
    --motion-file src/assets/motions/g1/<motion>.npz \
    --video-length 3200 --video-path render.mp4 \
    --video-width 640 --video-height 360

The video is written to ``--video-path`` (default policy_render.mp4).
"""
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mediapy as media
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass
class RenderConfig:
  checkpoint_file: str | None = None
  motion_file: str | None = None
  video_length: int = 3200
  video_path: str = "policy_render.mp4"
  video_width: int = 640
  video_height: int = 360
  num_envs: int = 1
  device: str | None = None


def main() -> None:
  # Import tasks to populate the registry (same pattern as play.py).
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    RenderConfig,
    args=remaining_args,
    default=RenderConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  if args.checkpoint_file is None:
    raise SystemExit("--checkpoint-file is required")

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(chosen_task, play=True)
  agent_cfg = load_rl_cfg(chosen_task)

  if args.motion_file is not None:
    motion_path = Path(args.motion_file)
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_path}")

  env_cfg.scene.num_envs = args.num_envs
  env_cfg.viewer.height = args.video_height
  env_cfg.viewer.width = args.video_width

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint_file, load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=device)

  fps = int(round(1.0 / raw_env.step_dt))
  obs = env.get_observations().to(device)
  frames: list = []
  print(f"[INFO] Rendering {args.video_length} frames -> {args.video_path}")
  for i in range(args.video_length):
    actions = policy(obs)
    obs, rewards, dones, extras = env.step(actions.to(device))
    frame = raw_env.render()
    if frame is not None:
      rgb = frame[0] if frame.ndim == 4 else frame
      frames.append(rgb)
    if (i + 1) % 500 == 0 or (i + 1) == args.video_length:
      print(f"  rendered {i + 1}/{args.video_length}")

  media.write_video(args.video_path, frames, fps=fps)
  raw_env.close()
  print(
    f"[INFO] Done. Video saved to {args.video_path} "
    f"({len(frames)} frames @ {fps} fps)"
  )


if __name__ == "__main__":
  main()
