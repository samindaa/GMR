"""Headless, parallel batch retargeting of LAFAN1 BVH motions to a robot.

Mirrors scripts/bvh_to_robot.py but runs many motions in parallel with offscreen
video rendering (no on-screen MuJoCo viewer). Saves one .pkl and one .mp4 per
input BVH under --output_dir, keeping the same filename stem.

    datasets/lafan1/walk1_subject1.bvh
      -> datasets/lafan1_output/walk1_subject1.pkl / .mp4

IMPORTANT: the saved `root_rot` is in **wxyz** (scalar-first) order, matching
MuJoCo's qpos convention. (Note this differs from scripts/bvh_to_robot.py, which
re-orders to xyzw.)

Motion selection (-m/--motions): one or more regexes matched (case-insensitive,
re.search) against each BVH file stem. A file is processed if ANY pattern
matches. Omit to process every motion.

Examples:
    # only walk and run motions, 6 workers
    python scripts/lafan1_bvh_to_robot_batch.py \
        --input_dir datasets/lafan1 \
        --output_dir datasets/lafan1_output \
        --robot unitree_g1 --workers 6 -m walk run
"""

# Offscreen GL backend so rendering works headless and in worker processes.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mujoco as mj

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def _make_camera(model, robot_base, distance):
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(robot_base).id
    cam.distance = distance
    cam.elevation = -10.0
    cam.azimuth = 90.0
    return cam


def process_one(bvh_file, input_dir, output_dir, robot, fmt, motion_fps,
                video, video_width, video_height):
    """Retarget a single BVH file; write .pkl (root_rot wxyz) and optional .mp4.

    Output path mirrors bvh_file's location relative to input_dir.
    """
    import imageio.v2 as imageio

    bvh_path = pathlib.Path(bvh_file)
    rel = bvh_path.relative_to(input_dir)
    out_base = pathlib.Path(output_dir) / rel.parent / rel.stem
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pkl_path = str(out_base) + ".pkl"
    mp4_path = str(out_base) + ".mp4"

    frames, human_height = load_bvh_file(str(bvh_path), format=fmt)

    retargeter = GMR(
        src_human=f"bvh_{fmt}", tgt_robot=robot,
        actual_human_height=human_height, verbose=False,
    )

    renderer = writer = cam = None
    if video:
        renderer = mj.Renderer(retargeter.model, height=video_height, width=video_width)
        cam = _make_camera(retargeter.model, ROBOT_BASE_DICT[robot],
                           VIEWER_CAM_DISTANCE_DICT[robot])
        writer = imageio.get_writer(mp4_path, fps=motion_fps)

    qpos_list = []
    for frame in frames:
        qpos = retargeter.retarget(frame)
        qpos_list.append(qpos.copy())
        if video:
            renderer.update_scene(retargeter.configuration.data, camera=cam)
            writer.append_data(renderer.render())

    if video:
        writer.close()
        renderer.close()

    qpos_arr = np.array(qpos_list)
    motion_data = {
        "fps": motion_fps,
        "root_pos": qpos_arr[:, :3],
        # keep MuJoCo scalar-first wxyz order (do NOT reorder to xyzw)
        "root_rot": qpos_arr[:, 3:7],
        "dof_pos": qpos_arr[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(motion_data, f)

    return f"{rel} -> {len(qpos_list)} frames -> {pkl_path}" + (
        f", {mp4_path}" if video else "")


def select_motions(input_dir, patterns):
    """Return .bvh files under input_dir whose stem matches any regex (or all)."""
    files = sorted(pathlib.Path(input_dir).rglob("*.bvh"))
    if not patterns:
        return files, []
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    matched, used = [], set()
    for f in files:
        for i, rx in enumerate(regexes):
            if rx.search(f.stem):
                matched.append(f)
                used.add(i)
                break
    unmatched = [patterns[i] for i in range(len(patterns)) if i not in used]
    return matched, unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory of LAFAN1 .bvh files.")
    parser.add_argument("--output_dir", required=True,
                        help="Output root; input subdir layout is mirrored here.")
    parser.add_argument("-m", "--motions", nargs="+", default=None,
                        help="Regexes (case-insensitive, search) on file stems; "
                             "omit to process all. e.g. -m walk run")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel worker processes (4-8 typical).")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video rendering (pkl only).")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    args = parser.parse_args()

    input_dir = str(pathlib.Path(args.input_dir).resolve())

    matched, unmatched = select_motions(input_dir, args.motions)
    if unmatched:
        print(f"[warn] no file matched: {unmatched}")
    if not matched:
        raise SystemExit("No .bvh files matched; nothing to do.")

    video = not args.no_video
    print(f"Processing {len(matched)} motions with {args.workers} workers "
          f"(video={'off' if args.no_video else 'on'}, root_rot=wxyz).")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, str(bvh), input_dir, args.output_dir,
                      args.robot, args.format, args.motion_fps, video,
                      args.video_width, args.video_height): bvh
            for bvh in matched
        }
        for fut in as_completed(futures):
            bvh = futures[fut]
            try:
                print("[done]", fut.result())
            except Exception as e:
                print(f"[FAILED] {bvh}: {e}")


if __name__ == "__main__":
    main()
