# robust_fsad

Official code for our BMVC paper **"Adversarially Robust Few-Shot Anomaly
Detection with Vision Foundation Models"**.

[[Paper]](<PAPER_LINK>) · [[arXiv]](<https://arxiv.org/abs/2510.13643>)

This repository evaluates the adversarial robustness of patch-based few-shot anomaly detection (DINOv2 backbone + k-NN memory bank) under white-box FGSM and PGD attacks, and provides two defenses:

- **PatchShift** – scores each test image over multiple sub-patch shifts and spatially filters the resulting patch-distance map.
- **FeaturePurifier** – a lightweight MLP trained on adversarial/clean feature pairs from the support set, applied to both memory-bank and test features.

Gradient-based attacks use a differentiable **DistanceProbe**, a surrogate MLP trained to regress per-patch k-NN distances to the memory bank. Trained probes are cached and reused across runs.

---

## 1. Environment setup

Tested with Python 3.10, PyTorch 2.4, CUDA 12.4 and FAISS-GPU 1.8.0.

```bash
conda create -n robust_fsad python=3.10 -y
conda activate robust_fsad
conda install -c pytorch -c nvidia pytorch=2.4 torchvision pytorch-cuda=12.4 faiss-gpu=1.8.0 cuda-toolkit=12.4 -y
pip install -r requirements.txt
```

DINOv2 weights are downloaded automatically via `torch.hub` (`facebookresearch/dinov2`) on first use.

---

## 2. File structure

```
robust_fsad/
├── run_defense.py              # Main entry point: memory bank, attack, defense, image-level evaluation
├── eval_pixel_level.py         # Post-hoc pixel-level AUROC / AUPRO from saved patch-distance maps
├── requirements.txt
├── src/
│   ├── backbones.py            # DINOv2 and torchvision ViT wrappers (feature extraction, masking)
│   ├── adversarial.py          # DistanceProbe surrogate; FGSM and PGD attacks
│   ├── distance_probe_cache.py # Train-or-load cache for DistanceProbe weights
│   ├── defense.py              # PatchShiftDefense, FeaturePurifier, apply_defenses_to_score
│   ├── post_eval.py            # Classification and segmentation metrics
│   └── utils.py                # Dataset metadata, augmentation, distance-map upsampling
├── data/                       # Datasets go here (see Section 3)
├── probe_cache/                # Created automatically: cached DistanceProbe weights
└── results_<DATASET>/          # Created automatically: anomaly maps, metrics, timings
```

---

## 3. Datasets

Place all datasets under `data/`. Every dataset must follow the MVTec-AD layout:

```
<data_root>/
├── <object>/
│   ├── train/
│   │   └── good/               # anomaly-free reference images
│   ├── test/
│   │   ├── good/
│   │   └── <defect_type>/      # one folder per defect type
│   └── ground_truth/
│       └── <defect_type>/      # pixel-level masks
└── ...
```

Expected locations:

```
data/
├── mvtec_anomaly_detection/    # MVTec AD
├── VisA_pytorch/
│   └── 1cls/                   # VisA, 1-class split
└── MPDD/                       # MPDD
```

| Dataset | Categories | Image / mask naming | Download |
|---|---|---|---|
| MVTec AD | 15 | `<id>.png` / `<id>_mask.png` | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| VisA | 12 | `<id>.JPG` / `<id>.png` | https://github.com/amazon-science/spot-diff |
| MPDD | 6 | `<id>.png` / `<id>_mask.png` | https://github.com/stepanje/MPDD |

### MVTec AD
Download and extract into `data/`. The archive already has the required layout:

```bash
mkdir -p data/mvtec_anomaly_detection
tar -xf mvtec_anomaly_detection.tar.xz -C data/mvtec_anomaly_detection
```

### VisA
VisA must be converted to the official 1-class split with the script provided in the VisA repository:

```bash
git clone https://github.com/amazon-science/spot-diff.git
cd spot-diff
wget https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
mkdir VisA && tar -xf VisA_20220922.tar -C VisA
python ./utils/prepare_data.py --split-type 1cls --data-folder ./VisA \
       --save-folder ./VisA_pytorch --split-file ./split_csv/1cls.csv
mv VisA_pytorch <path_to>/robust_fsad/data/
```

### MPDD
Download from the link in the MPDD repository and extract to `data/MPDD/` so that each of the six categories (`bracket_black`, `bracket_brown`, `bracket_white`, `connector`, `metal_plate`, `tubes`) follows the layout above. Defect sub-types are discovered from `test/` automatically.

---

## 4. Usage

### 4.1 Clean baseline (no attack, no defense)

```bash
python run_defense.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --shots 4 --attack_type none --defense none --device cuda:0
```

### 4.2 Attack without defense

```bash
# FGSM, eps = 8/255
python run_defense.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --shots 4 --attack_type fgsm --epsilon 0.03137 --defense none --device cuda:0

# PGD-20, eps = 8/255
python run_defense.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --shots 4 --attack_type pgd --epsilon 0.03137 --pgd_iters 20 --defense none --device cuda:0
```

### 4.3 Attack with defense

```bash
# FGSM attack; PatchShift + FeaturePurifier (purifier trained on FGSM pairs)
python run_defense.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --shots 4 --attack_type fgsm --epsilon 0.03137 \
    --defense patchshift_purifier --purifier_attack fgsm --device cuda:0

# PGD-20 attack; PatchShift + FeaturePurifier (purifier trained on PGD-10 pairs)
python run_defense.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --shots 4 --attack_type pgd --epsilon 0.03137 --pgd_iters 20 \
    --defense patchshift_purifier --purifier_attack pgd --purifier_pgd_iters 10 --device cuda:0
```

Use `--defense patchshift` or `--defense purifier` to evaluate each component alone.

### 4.4 VisA and MPDD

```bash
python run_defense.py --dataset VisA --data_root data/VisA_pytorch/1cls \
    --shots 4 --attack_type pgd --epsilon 0.03137 --pgd_iters 20 \
    --defense patchshift_purifier --purifier_attack pgd --device cuda:0

python run_defense.py --dataset MPDD --data_root data/MPDD \
    --shots 4 --attack_type pgd --epsilon 0.03137 --pgd_iters 20 \
    --defense patchshift_purifier --purifier_attack pgd --device cuda:0
```

### 4.5 Useful options

| Argument | Default | Description |
|---|---|---|
| `--shots` | `4` | Number of reference (support) images per category |
| `--num_seeds` / `--just_seed` | `1` / `None` | Number of support-set draws, or run a single seed |
| `--objects` | all | Restrict to specific categories, e.g. `--objects bottle screw` |
| `--attack_target_samples` | `all` | Attack `all`, `normal`, or `anomaly` test samples |
| `--epsilon` | `8/255` | L∞ perturbation budget |
| `--n_shifts`, `--max_shift` | `8`, `7` | PatchShift settings |
| `--purifier_epochs` | `100` | FeaturePurifier training epochs |
| `--probe_n_train_images` | `50` | Held-out normal images for DistanceProbe training |
| `--probe_cache_root` | `probe_cache` | Cache location for trained probes |
| `--probe_force_retrain` | `False` | Ignore cache and retrain the probe |
| `--faiss_on_cpu` | `False` | Use CPU FAISS index |
| `--tag` | `None` | Suffix appended to the results directory name |

Run `python run_defense.py --help` for the full list.

### 4.6 Pixel-level evaluation

`run_defense.py` reports image-level metrics (AUROC, AP, F1) and saves patch-distance maps as `.npy`. Pixel-level AUROC and AUPRO are computed afterwards:

```bash
python eval_pixel_level.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --results_dir <results_dir> --seeds 0

# all seeds present in the results directory, with mean/std aggregation
python eval_pixel_level.py --dataset MVTec --data_root data/mvtec_anomaly_detection \
    --results_dir <results_dir> --seeds_auto
```

### 4.7 Outputs

Results are written to

```
results_<DATASET>/<model>_<resolution>/<shots>-shot_preprocess=<p>_attack=<a>_eps=<eps>_defense=<d>/
├── args.yaml
├── metrics_seed=<s>.json
├── time_measurements.csv
└── anomaly_maps/seed=<s>/<object>/test/<defect_type>/*.npy
```

---

## Citation

If you find this code useful, please cite:

```bibtex
<BIBTEX ENTRY>
```

---

## Acknowledgements

This code is built on top of [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025). We thank the authors for making their code publicly available.

```bibtex
@inproceedings{damm2024anomalydino,
  title     = {AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2},
  author    = {Simon Damm and Mike Laszkiewicz and Johannes Lederer and Asja Fischer},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV 2025)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2405.14529}
}
```

We also thank the authors of [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff) and [MPDD](https://github.com/stepanje/MPDD) for releasing their datasets.
