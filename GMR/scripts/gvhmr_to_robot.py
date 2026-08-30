import argparse
import pathlib
import os
import time
import numpy as np
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
from rich import print

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr_pred_file", type=str, required=True,
                        help="SMPLX motion file to load.")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite",
                 "openloong", "tienkung"],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion.")
    parser.add_argument("--loop", default=False, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument("--rate_limit", default=False, action="store_true")
    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    print("[1/4] Loading GVHMR prediction...")
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )

    print("[2/4] Aligning FPS...")
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    print(f"    frames: {len(smplx_data_frames)}, fps: {aligned_fps}")

    print("[3/4] Initializing GMR...")
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = None
    if args.record_video:
        from general_motion_retargeting import RobotMotionViewer
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=True,
            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",
        )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    print("[4/4] Retargeting...")

    # Setup MuJoCo for ground alignment
    import mujoco
    import general_motion_retargeting.params as _params
    _xml_path = str(_params.ROBOT_XML_DICT[args.robot])
    _mj_model = mujoco.MjModel.from_xml_path(_xml_path)
    _mj_data = mujoco.MjData(_mj_model)
    _lf_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_toe_link")
    _rf_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_toe_link")
    GROUND_TARGET = 0.02

    num_frames = len(smplx_data_frames)
    for i in range(num_frames):
        smplx_data = smplx_data_frames[i]
        qpos = retarget.retarget(smplx_data, offset_to_ground=True)

        # Post-process: adjust root_pos Z so foot touches ground
        _mj_data.qpos[:3] = qpos[:3]
        _mj_data.qpos[3:7] = qpos[3:7]
        _mj_data.qpos[7:] = qpos[7:]
        mujoco.mj_forward(_mj_model, _mj_data)
        foot_min_z = min(_mj_data.xpos[_lf_id][2], _mj_data.xpos[_rf_id][2])
        qpos[2] += GROUND_TARGET - foot_min_z

        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )

        if args.save_path is not None:
            qpos_list.append(qpos)

        if (i + 1) % 50 == 0 or (i + 1) == num_frames:
            print(f"    processed {i+1}/{num_frames}")

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()

    if args.save_path is not None:
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
        print(f"  frames: {len(qpos_list)}, dof shape: {dof_pos.shape}")
