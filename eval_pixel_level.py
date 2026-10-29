"""
Post-hoc pixel-level AUROC/PRO evaluation.

Reads .npy patch-distance maps saved by run_defense.py, upsamples them to
full image resolution via dists2map, writes .tiff files, and reruns
eval_finished_run with eval_segm=True. Per-seed metrics are merged with
the existing classification metrics.

Single seed:
    python eval_pixel_level.py \
        --results_dir <results_dir> \
        --data_root data/mvtec_anomaly_detection \
        --seeds 0

Multiple seeds:
    python eval_pixel_level.py --results_dir <results_dir> --seeds 0 1 2

Auto-detect all seeds present in the results dir:
    python eval_pixel_level.py --results_dir <results_dir> --seeds_auto
"""

import argparse
import os
import glob
import json
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

from src.utils import dists2map
from src.post_eval import eval_finished_run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True,
                   help="The results_dir produced by run_defense.py "
                        "(contains anomaly_maps/seed=N/<obj>/test/<type>/<img>.npy)")
    p.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection")
    p.add_argument("--dataset", type=str, default="MVTec",
                   choices=["MVTec", "VisA", "MPDD"])
    p.add_argument("--seeds", type=int, nargs='+', default=None,
                   help="One or more seeds to evaluate, e.g. --seeds 0 1 2. "
                        "Mutually exclusive with --seeds_auto.")
    p.add_argument("--seeds_auto", action="store_true",
                   help="Auto-detect every seed=N directory under "
                        "<results_dir>/anomaly_maps/ and process all of them.")
    p.add_argument("--pro_integration_limit", type=float, default=0.3)
    p.add_argument("--keep_tiff", action="store_true",
                   help="Do not delete .tiff files after evaluation")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip seeds whose merged metrics already include "
                        "'mean_segmentation_au_roc'.")
    p.add_argument("--objects", type=str, nargs='+', default=None,
               help="Restrict evaluation to these categories.")
    return p.parse_args()


def npy_to_tiff(npy_path, tiff_path, target_shape):
    """Convert a patch-grid distance map to a full-resolution anomaly map."""
    d_grid = np.load(npy_path).astype(np.float32)
    if d_grid.ndim == 1:
        n = d_grid.shape[0]
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError(
                "Cannot infer grid shape from flat array of size {} at {}".format(
                    n, npy_path))
        d_grid = d_grid.reshape(side, side)
    anomaly_map = dists2map(d_grid, target_shape)
    tiff.imwrite(tiff_path, anomaly_map)


def discover_seeds(results_dir):
    """Find every seed=N directory under <results_dir>/anomaly_maps/."""
    base = os.path.join(results_dir, 'anomaly_maps')
    if not os.path.isdir(base):
        return []
    seeds = []
    for entry in sorted(os.listdir(base)):
        if entry.startswith('seed=') and os.path.isdir(os.path.join(base, entry)):
            try:
                seeds.append(int(entry.split('=', 1)[1]))
            except ValueError:
                continue
    return sorted(seeds)


def process_seed(args, seed):
    """Convert .npy -> .tiff for one seed and run pixel-level eval."""
    anomaly_maps_dir = os.path.join(args.results_dir,
                                    'anomaly_maps',
                                    'seed={}'.format(seed))
    if not os.path.isdir(anomaly_maps_dir):
        print("[seed {}] anomaly_maps dir not found: {}; skipping.".format(
            seed, anomaly_maps_dir))
        return False

    metric_file = os.path.join(args.results_dir,
                               'metrics_seed={}.json'.format(seed))

    # Optional: skip if pixel-level metrics are already present.
    if args.skip_existing and os.path.isfile(metric_file):
        try:
            with open(metric_file) as f:
                m = json.load(f)
            if 'mean_segmentation_au_roc' in m:
                print("[seed {}] Pixel-level metrics already present; skipping "
                      "(--skip_existing).".format(seed))
                return True
        except Exception:
            pass

    npy_paths = sorted(glob.glob(os.path.join(
        anomaly_maps_dir, '*', 'test', '*', '*.npy')))
    if len(npy_paths) == 0:
        print("[seed {}] No .npy anomaly maps under {}; skipping.".format(
            seed, anomaly_maps_dir))
        return False

    print("\n=========== seed={} ===========".format(seed))
    print("Converting {} .npy maps -> .tiff ...".format(len(npy_paths)))
    for npy_path in tqdm(npy_paths, desc="seed={}".format(seed), leave=False):
        rel = os.path.relpath(npy_path, anomaly_maps_dir)
        parts = rel.split(os.sep)
        if len(parts) != 4 or parts[1] != 'test':
            continue
        obj, _, type_anomaly, fname = parts
        stem = os.path.splitext(fname)[0]

        img_dir = os.path.join(args.data_root, obj, 'test', type_anomaly)
        cand = None
        for ext in ('.png', '.PNG', '.jpg', '.JPG', '.jpeg'):
            cp = os.path.join(img_dir, stem + ext)
            if os.path.isfile(cp):
                cand = cp
                break
        if cand is None:
            raise FileNotFoundError(
                "Could not find test image for {} under {}".format(stem, img_dir))
        img = np.array(Image.open(cand))
        target_shape = img.shape[:2]

        tiff_path = os.path.splitext(npy_path)[0] + '.tiff'
        npy_to_tiff(npy_path, tiff_path, target_shape)

    existing = {}
    if os.path.isfile(metric_file):
        with open(metric_file) as f:
            existing = json.load(f)
        backup = metric_file + '.bak'
        if not os.path.isfile(backup):
            with open(backup, 'w') as f:
                json.dump(existing, f, indent=4)
            print("[seed {}] Backed up existing metrics to {}".format(seed, backup))

    print("[seed {}] Running pixel-level evaluation ...".format(seed))
    eval_finished_run(
        dataset=args.dataset,
        dataset_base_dir=args.data_root,
        anomaly_maps_dir=anomaly_maps_dir,
        output_dir=args.results_dir,
        seed=seed,
        objects=args.objects,
        pro_integration_limit=args.pro_integration_limit,
        eval_clf=False,
        eval_segm=True,
        delete_tiff_files=not args.keep_tiff,
    )

    if existing:
        with open(metric_file) as f:
            seg_metrics = json.load(f)
        merged = {}
        all_keys = set(existing.keys()) | set(seg_metrics.keys())
        for k in all_keys:
            if isinstance(existing.get(k), dict) and isinstance(seg_metrics.get(k), dict):
                merged[k] = {**existing[k], **seg_metrics[k]}
            elif k in seg_metrics:
                merged[k] = seg_metrics[k]
            else:
                merged[k] = existing[k]
        with open(metric_file, 'w') as f:
            json.dump(merged, f, indent=4)
        print("[seed {}] Merged metrics written to {}".format(seed, metric_file))
    return True


def aggregate_across_seeds(results_dir, seeds):
    """Print mean and std of headline metrics across seeds when available."""
    keys_of_interest = (
        'mean_classification_au_roc',
        'mean_segmentation_au_roc',
        'mean_au_pro',
    )
    per_metric = {k: [] for k in keys_of_interest}
    for s in seeds:
        f = os.path.join(results_dir, 'metrics_seed={}.json'.format(s))
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            m = json.load(fh)
        for k in keys_of_interest:
            if k in m and isinstance(m[k], (int, float)):
                per_metric[k].append((s, float(m[k])))

    print("\n=========== Aggregate across seeds ===========")
    any_data = False
    for k, pairs in per_metric.items():
        if not pairs:
            continue
        any_data = True
        seeds_with = [s for s, _ in pairs]
        vals = np.array([v for _, v in pairs], dtype=np.float64)
        print("{:<32s} seeds={}  n={}  mean={:.4f}  std={:.4f}  "
              "min={:.4f}  max={:.4f}".format(
                  k, seeds_with, vals.size, vals.mean(),
                  vals.std(ddof=0), vals.min(), vals.max()))
    if not any_data:
        print("(no headline metrics found across seeds)")


def main():
    args = parse_args()

    if args.seeds_auto and args.seeds is not None:
        raise SystemExit("Use either --seeds or --seeds_auto, not both.")
    if args.seeds_auto:
        seeds = discover_seeds(args.results_dir)
        if not seeds:
            raise SystemExit("No seed=N directories under {}".format(
                os.path.join(args.results_dir, 'anomaly_maps')))
        print("Auto-detected seeds: {}".format(seeds))
    elif args.seeds is not None:
        seeds = sorted(set(args.seeds))
    else:
        seeds = [0]  # backward-compat default

    succeeded = []
    for s in seeds:
        if process_seed(args, s):
            succeeded.append(s)

    if len(succeeded) >= 2:
        aggregate_across_seeds(args.results_dir, succeeded)

    print("\nDone. Processed seeds: {}".format(succeeded))


if __name__ == "__main__":
    main()