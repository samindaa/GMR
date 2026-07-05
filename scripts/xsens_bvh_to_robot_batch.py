"""Headless, parallel batch retargeting of Xsens/100STYLE BVH motions to a robot.

Renders video offscreen (no on-screen MuJoCo viewer) so many motions can be
processed in parallel. Saves one .pkl and one .mp4 per input BVH, mirroring the
input directory layout under --output_dir (same subdir + same filename stem).

    datasets/100STYLE/Neutral/Neutral_FW.bvh
      -> datasets/100STYLE_output/Neutral/Neutral_FW.pkl / .mp4

Categories (-c) are regex patterns matched case-insensitively (full match)
against the sub-directory names of --input_dir.

Frame trimming: if a Frame_Cuts.csv is found (default <input_dir>/Frame_Cuts.csv),
each motion is clipped to its [<SUFFIX>_START, <SUFFIX>_STOP] range, where the
style is the file prefix and SUFFIX is the part after the last underscore
(e.g. Neutral_FW -> style "Neutral", suffix "FW"). N/A or missing -> no clip.

Examples:
    python scripts/xsens_bvh_to_robot_batch.py \
        --input_dir datasets/100STYLE \
        --output_dir datasets/100STYLE_output \
        --robot unitree_g1 --workers 8 \
        -c Aeroplane Chicken CrossOver Dinosaur FlickLegs HandsBetweenLegs \
           HighKnees Neutral Skip SpinClock Superman Zombie
"""

# Offscreen GL backend so rendering works headless and in worker processes.
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
import pathlib
import pickle
import re
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mujoco as mj

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.xsens import load_xsens_file


def load_frame_cuts(csv_path):
    """Parse Frame_Cuts.csv into {style_lower: {suffix: (start, stop)}}.

    stop is stored as an exclusive end (None if N/A / missing).
    """
    cuts = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # columns after STYLE_NAME come in <SUFFIX>_START, <SUFFIX>_STOP pairs
        suffixes = [h[:-6] for h in header[1:] if h.endswith("_START")]
        for row in reader:
            if not row or not row[0].strip():
                continue
            style = row[0].strip()
            fields = dict(zip(header, row))
            per = {}
            for suf in suffixes:
                s, e = fields.get(f"{suf}_START"), fields.get(f"{suf}_STOP")
                if s in (None, "", "N/A") or e in (None, "", "N/A"):
                    continue
                per[suf] = (int(s), int(e))
            cuts[style.lower()] = per
    return cuts


def _make_camera(model, robot_base, distance):
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(robot_base).id
    cam.distance = distance
    cam.elevation = -10.0
    cam.azimuth = 90.0
    return cam


def process_one(bvh_file, input_dir, output_dir, robot, scale, video,
                video_width, video_height, start, end):
    """Retarget a single BVH file; write .pkl and (optionally) .mp4.

    Output path mirrors bvh_file's location relative to input_dir.
    """
    import imageio.v2 as imageio

    bvh_path = pathlib.Path(bvh_file)
    rel = bvh_path.relative_to(input_dir)
    out_base = pathlib.Path(output_dir) / rel.parent / rel.stem
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pkl_path = str(out_base) + ".pkl"
    mp4_path = str(out_base) + ".mp4"

    load_args = types.SimpleNamespace(
        bvh_file=str(bvh_path), scale=scale, start=start, end=end,
        reset_to_zero=False, bvh_format="3DSM",
    )
    frames, human_height, frame_time = load_xsens_file(load_args)
    motion_fps = int(1 / frame_time)

    retargeter = GMR(
        src_human="bvh_xsens", tgt_robot=robot,
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
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "fps": motion_fps,
            "root_pos": qpos_arr[:, :3],
            "root_rot": qpos_arr[:, 3:7],
            "dof_pos": qpos_arr[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }, f)

    cut = "" if start is None else f" [cut {start}:{end}]"
    return f"{rel} -> {len(qpos_list)} frames{cut} -> {pkl_path}" + (
        f", {mp4_path}" if video else "")


def select_categories(input_dir, patterns):
    """Return subdir names of input_dir that fully match any regex pattern."""
    dirs = sorted(p.name for p in pathlib.Path(input_dir).iterdir() if p.is_dir())
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    matched, used = [], set()
    for d in dirs:
        for i, rx in enumerate(regexes):
            if rx.fullmatch(d):
                matched.append(d)
                used.add(i)
                break
    unmatched = [patterns[i] for i in range(len(patterns)) if i not in used]
    return matched, unmatched


def frame_cut_for(stem, cuts):
    """Look up (start, end) for a file stem like 'Neutral_FW'."""
    if "_" not in stem:
        return None, None
    style, suffix = stem.rsplit("_", 1)
    se = cuts.get(style.lower(), {}).get(suffix)
    return (se[0], se[1]) if se else (None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Dataset root containing category subdirectories.")
    parser.add_argument("--output_dir", required=True,
                        help="Output root; input subdir layout is mirrored here.")
    parser.add_argument("-c", "--categories", nargs="+", required=True,
                        help="Category names or regexes (case-insensitive, full match).")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--frame_cuts", default=None,
                        help="Path to Frame_Cuts.csv (default: <input_dir>/Frame_Cuts.csv).")
    parser.add_argument("--no_frame_cuts", action="store_true",
                        help="Do not trim motions with Frame_Cuts.csv.")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip video rendering (pkl only).")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    args = parser.parse_args()

    input_dir = str(pathlib.Path(args.input_dir).resolve())

    # Frame cuts
    cuts = {}
    if not args.no_frame_cuts:
        cuts_path = args.frame_cuts or os.path.join(input_dir, "Frame_Cuts.csv")
        if os.path.exists(cuts_path):
            cuts = load_frame_cuts(cuts_path)
            print(f"Loaded frame cuts for {len(cuts)} styles from {cuts_path}")
        else:
            print(f"[warn] Frame_Cuts.csv not found at {cuts_path}; processing full motions.")

    # Category selection
    matched, unmatched = select_categories(input_dir, args.categories)
    if unmatched:
        print(f"[warn] no directory matched: {unmatched}")
    if not matched:
        raise SystemExit("No categories matched; nothing to do.")
    print(f"Matched categories ({len(matched)}): {matched}")

    # Collect files + per-file cuts
    jobs = []
    for cat in matched:
        for bvh in sorted(pathlib.Path(input_dir, cat).glob("*.bvh")):
            start, end = frame_cut_for(bvh.stem, cuts)
            jobs.append((str(bvh), start, end))
    if not jobs:
        raise SystemExit("No .bvh files found under matched categories.")

    video = not args.no_video
    print(f"Processing {len(jobs)} motions with {args.workers} workers "
          f"(video={'off' if args.no_video else 'on'}).")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, bvh, input_dir, args.output_dir, args.robot,
                      args.scale, video, args.video_width, args.video_height,
                      start, end): bvh
            for bvh, start, end in jobs
        }
        for fut in as_completed(futures):
            bvh = futures[fut]
            try:
                print("[done]", fut.result())
            except Exception as e:
                print(f"[FAILED] {bvh}: {e}")


if __name__ == "__main__":
    main()
