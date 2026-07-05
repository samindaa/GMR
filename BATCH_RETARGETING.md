# Batch Retargeting (Headless, Parallel)

Two helper scripts retarget whole datasets to a robot **without opening the
MuJoCo viewer**, rendering each motion's video offscreen and running several
motions in parallel:

| Script | Source format | Dataset |
| --- | --- | --- |
| [`scripts/xsens_bvh_to_robot_batch.py`](scripts/xsens_bvh_to_robot_batch.py) | Xsens BVH (`bvh_xsens`) | 100STYLE |
| [`scripts/lafan1_bvh_to_robot_batch.py`](scripts/lafan1_bvh_to_robot_batch.py) | LAFAN1 BVH (`bvh_lafan1`) | LAFAN1 |

Each input `.bvh` produces a `.pkl` (retargeted robot motion) and, unless
disabled, an `.mp4` (rendered video). Output mirrors the input directory layout
and keeps the same filename stem.

## Requirements

- The `gmr` conda environment (see [README](README.md)) with the robot assets.
- Offscreen rendering uses EGL; the scripts set `MUJOCO_GL=egl` automatically.
  If EGL is unavailable on your machine, run with `MUJOCO_GL=osmesa` instead:
  ```bash
  MUJOCO_GL=osmesa python scripts/lafan1_bvh_to_robot_batch.py ...
  ```
- Pick `--workers` to match your machine (4–8 is typical). More workers = more
  RAM/GPU use.

## Output `.pkl` format

Both scripts save a pickled dict:

```python
{
    "fps": int,
    "root_pos": (N, 3) float,     # base translation
    "root_rot": (N, 4) float,     # base orientation quaternion
    "dof_pos":  (N, D) float,     # joint positions
    "local_body_pos": None,
    "link_body_list": None,
}
```

> **Quaternion order differs by script — read this before consuming the data.**
> - `lafan1_bvh_to_robot_batch.py` saves `root_rot` in **wxyz** (scalar-first,
>   MuJoCo convention).
> - `xsens_bvh_to_robot_batch.py` saves `root_rot` in **wxyz** as well (it stores
>   the raw MuJoCo `qpos[3:7]`).
>
> Note this is *not* the same as the single-file `scripts/bvh_to_robot.py`, which
> reorders `root_rot` to **xyzw** before saving.

---

## 1. 100STYLE / Xsens — `xsens_bvh_to_robot_batch.py`

Retargets 100STYLE categories, mirroring the `datasets/100STYLE/<Category>/`
layout and optionally trimming each clip using `Frame_Cuts.csv`.

```
datasets/100STYLE/Neutral/Neutral_FW.bvh
  -> datasets/100STYLE_output/Neutral/Neutral_FW.pkl / .mp4
```

### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--input_dir` | (required) | Dataset root containing category subdirectories. |
| `--output_dir` | (required) | Output root; input subdir layout is mirrored here. |
| `-c`, `--categories` | (required) | Category names/regexes, **case-insensitive full match** against subdir names. |
| `--robot` | `unitree_g1` | Target robot. |
| `--workers` | `4` | Parallel worker processes. |
| `--scale` | `0.01` | BVH unit scale passed to the loader. |
| `--frame_cuts` | `<input_dir>/Frame_Cuts.csv` | Path to the cut table. |
| `--no_frame_cuts` | off | Disable trimming; retarget full clips. |
| `--no_video` | off | Skip video rendering (pkl only, much faster). |
| `--video_width` / `--video_height` | `640` / `480` | Video resolution. |

### Category matching

`-c` values are **regexes matched against the directory names** (full match,
case-insensitive). The 100STYLE directory names differ from common spellings, so
use the exact ones:

| You might write | Actual directory |
| --- | --- |
| AeroPlane | `Aeroplane` |
| Dinasaur | `Dinosaur` |
| Flicklegs | `FlickLegs` |
| Spin (Clockwise) | `SpinClock` (also `SpinAntiClock`) |

### Frame cuts

If a `Frame_Cuts.csv` is found, each motion is clipped to
`[<SUFFIX>_START, <SUFFIX>_STOP]`, where the style is the file prefix and the
suffix is the part after the last underscore (e.g. `Neutral_FW` → style
`Neutral`, suffix `FW`). `N/A` or missing entries are left untrimmed.

### Working examples

Process a single category (Neutral), 4 workers:
```bash
cd /home/saminda/third_party/GMR
python scripts/xsens_bvh_to_robot_batch.py \
    --input_dir datasets/100STYLE \
    --output_dir datasets/100STYLE_output \
    --robot unitree_g1 --workers 4 \
    -c Neutral
```

Process the full 12-category set (99 motions), 10 workers:
```bash
python scripts/xsens_bvh_to_robot_batch.py \
    --input_dir datasets/100STYLE \
    --output_dir datasets/100STYLE_output \
    --robot unitree_g1 --workers 10 \
    -c Aeroplane Chicken CrossOver Dinosaur FlickLegs HandsBetweenLegs \
       HighKnees Neutral Skip SpinClock Superman Zombie
```

Regex selection — every "Spin" category, no trimming, pkl only:
```bash
python scripts/xsens_bvh_to_robot_batch.py \
    --input_dir datasets/100STYLE \
    --output_dir datasets/100STYLE_output \
    -c 'Spin.*' --no_frame_cuts --no_video --workers 4
```

---

## 2. LAFAN1 — `lafan1_bvh_to_robot_batch.py`

Retargets LAFAN1's flat set of `.bvh` files. Selection is by regex on the file
stem, so you can pick motion types like `walk` or `run`.

```
datasets/lafan1/walk1_subject1.bvh
  -> datasets/lafan1_output/walk1_subject1.pkl / .mp4
```

### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--input_dir` | (required) | Directory of LAFAN1 `.bvh` files. |
| `--output_dir` | (required) | Output root; input subdir layout is mirrored here. |
| `-m`, `--motions` | (all) | Regexes (**case-insensitive `re.search`**) on file stems; a file runs if any pattern matches. |
| `--robot` | `unitree_g1` | Target robot. |
| `--format` | `lafan1` | Source format (`lafan1` or `nokov`). |
| `--motion_fps` | `30` | Output FPS (LAFAN1 loader returns no frame time). |
| `--workers` | `6` | Parallel worker processes. |
| `--no_video` | off | Skip video rendering (pkl only). |
| `--video_width` / `--video_height` | `640` / `480` | Video resolution. |

### Motion matching

`-m` uses `re.search` (substring/regex), so `-m walk run` matches
`walk1_subject1`, `run2_subject5`, etc. Because it is a substring search it also
matches names that *contain* the term; anchor with `^` for an exact prefix:

```bash
-m '^walk' '^run'
```

### Working examples

Only walk and run, 6 workers:
```bash
cd /home/saminda/third_party/GMR
python scripts/lafan1_bvh_to_robot_batch.py \
    --input_dir datasets/lafan1 \
    --output_dir datasets/lafan1_output \
    --robot unitree_g1 --workers 6 \
    -m walk run
```

Every motion, 8 workers:
```bash
python scripts/lafan1_bvh_to_robot_batch.py \
    --input_dir datasets/lafan1 \
    --output_dir datasets/lafan1_output \
    --robot unitree_g1 --workers 8
```

Dance motions, pkl only (no video), faster:
```bash
python scripts/lafan1_bvh_to_robot_batch.py \
    --input_dir datasets/lafan1 \
    --output_dir datasets/lafan1_output \
    -m dance --no_video --workers 8
```

---

## Tips

- **Faster iteration:** add `--no_video` to produce only `.pkl` files, then
  render selectively later.
- **Progress & failures:** each finished motion prints a `[done]` line; failures
  print `[FAILED] <file>: <error>` and do not stop the rest of the batch.
- **Datasets are large:** generated `.pkl`/`.mp4` files are covered by
  `.gitignore`; do not commit the `*_output/` directories.
- **Verify one first:** for a new robot or config, run a single category/motion
  and check the `.mp4` before launching the full batch.
