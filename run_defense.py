"""
Usage example:
    python run_defense.py --dataset MVTec --shots 4 --attack_type fgsm --epsilon 0.03137 \
        --defense patchshift_purifier --purifier_attack fgsm --device cuda:0
    python run_defense.py --dataset MVTec --shots 4 --attack_type pgd --epsilon 0.03137 \
        --pgd_iters 20 --defense patchshift_purifier --purifier_attack pgd --purifier_pgd_iters 10 --device cuda:0
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
import faiss
import cv2
from tqdm import tqdm, trange
from PIL import Image
import tifffile as tiff
import yaml
from src.backbones import get_model
from src.utils import get_dataset_info, augment_image, dists2map
from src.post_eval import eval_finished_run
from src.adversarial import fgsm_trained_probe, pgd_trained_probe
from src.defense import (PatchShiftDefense,
                         FeaturePurifier, apply_defenses_to_score)
from src.distance_probe_cache import train_or_load_distance_probe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec",
                        choices=["MVTec", "VisA", "MPDD"])
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--preprocess", type=str, default="agnostic")
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup_iters", type=int, default=25)
    parser.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--objects", nargs='+', type=str, default=None)
    # Attack
    parser.add_argument("--attack_type", type=str, default="none",
                        choices=["none", "fgsm", "pgd"])
    parser.add_argument("--attack_target_samples", type=str, default="all")
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--pgd_iters", type=int, default=20,
                        help="Iterations for pgd (default 20)")
    # Defense
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "patchshift", "purifier",
                                 "patchshift_purifier"])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--n_shifts", type=int, default=8)
    parser.add_argument("--max_shift", type=int, default=7)
    parser.add_argument("--purifier_epochs", type=int, default=100)
    parser.add_argument("--purifier_lr", type=float, default=1e-3)
    parser.add_argument("--purifier_noise_std", type=float, default=0.03)
    parser.add_argument("--purifier_n_noise", type=int, default=4)
    parser.add_argument("--purifier_attack", type=str, default='fgsm',
                        choices=['fgsm', 'pgd'],
                        help="Attack used to generate adversarial training "
                             "pairs for the FeaturePurifier.")
    parser.add_argument("--purifier_pgd_iters", type=int, default=10,
                        help="PGD iterations when --purifier_attack is 'pgd'.")
    parser.add_argument("--probe_n_train_images", type=int, default=50,
                    help="Number of held-out normal train images to use "
                         "for DistanceProbe regression training. The probe "
                         "learns to predict per-patch k-NN distance to the "
                         "support memory bank, serving as a differentiable "
                         "proxy for the actual scorer used at attack time. "
                         "Larger N = more training signal at the cost of "
                         "drifting further from the few-shot threat model "
                         "(though the probe is only used by the attacker, "
                         "not the detector, so this is consistent with "
                         "white-box attacker access to training data).")
    parser.add_argument("--probe_epochs", type=int, default=100,
                        help="Training epochs for DistanceProbe.")
    parser.add_argument("--probe_hidden_dim", type=int, default=None,
                        help="MLP hidden dimension. Default: feat_dim.")
    parser.add_argument("--probe_cache_root", type=str, default='probe_cache',
                    help="Directory where trained DistanceProbe weights are "
                         "cached. The probe depends only on (backbone, "
                         "resolution, object, support indices, preprocessing, "
                         "probe hyperparams), so a single cache covers many "
                         "run_defense.py invocations. Default: ./probe_cache")
    parser.add_argument("--probe_force_retrain", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="If set, ignore any cached probe and retrain. Use "
                            "when the probe training code has changed in a way "
                            "the cache key cannot capture.")
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def run_defended_detection(model, object_name, data_root, n_ref_samples,
                           object_anomalies, plots_dir,
                           attack_type='none', attack_target_samples='all',
                           epsilon=8/255, pgd_iters=20,
                           patch_shift=None, feature_purifier=None,
                           masking=None, mask_ref_images=False,
                           rotation=False, knn_metric='L2_normalized',
                           knn_neighbors=1, faiss_on_cpu=False,
                           seed=0, save_patch_dists=True, save_tiffs=False, probe_n_train_images=50, probe_epochs=100, probe_hidden_dim=None):
    device = model.device
    type_anomalies = list(set(object_anomalies[object_name] + ['good']))

    features_ref = []
    support_images_raw = []
    img_ref_folder = os.path.join(data_root, object_name, 'train', 'good')
    if n_ref_samples == -1:
        img_ref_samples = sorted(os.listdir(img_ref_folder))
    else:
        img_ref_samples = sorted(os.listdir(img_ref_folder))[
            seed * n_ref_samples: (seed + 1) * n_ref_samples]

    start_time = time.time()
    for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
        img_path = os.path.join(img_ref_folder, img_ref_n)
        image_ref = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        support_images_raw.append(image_ref)
        imgs = augment_image(image_ref) if rotation else [image_ref]
        for img in imgs:
            tensor, grid = model.prepare_image(img)
            feats = model.extract_features(tensor)
            bg_mask = model.compute_background_mask(feats, grid, threshold=10,
                                                    masking_type=(mask_ref_images and masking))
            features_ref.append(feats[bg_mask])

    features_ref = np.concatenate(features_ref, axis=0).astype('float32')
    features_ref_raw = features_ref.copy()

    # Train surrogate probe (needed for all gradient-based attacks AND purifier training)
    probe = None
    gradient_attacks = ('fgsm','pgd')
    if attack_type in gradient_attacks or feature_purifier is not None:    
        probe = train_or_load_distance_probe(
                model, features_ref_raw,
                masking=masking, mask_ref_images=mask_ref_images,
                rotation=rotation,
                data_root=data_root, object_name=object_name,
                n_train_images=probe_n_train_images,
                epochs=probe_epochs,
                hidden_dim=probe_hidden_dim,
                device=device,
                # Cache-control args
                backbone_name=args.model_name,
                resolution=args.resolution,
                shots=n_ref_samples,
                support_seed=seed,
                cache_root=args.probe_cache_root,
                force_retrain=args.probe_force_retrain)

    if feature_purifier is not None:
        feat_dim = features_ref_raw.shape[1]
        feature_purifier.feat_dim = feat_dim
        feature_purifier.train_purifier(
            model, support_images_raw, masking, mask_ref_images,
            rotation=rotation, probe=probe, epsilon=epsilon)
        print("  Purifying memory bank features...")
        features_ref = feature_purifier.purify(features_ref_raw)

    features_ref_index = features_ref.copy().astype('float32')
    if faiss_on_cpu:
        knn_index = faiss.IndexFlatL2(features_ref_index.shape[1])
    else:
        res = faiss.StandardGpuResources()
        knn_index = faiss.GpuIndexFlatL2(res, features_ref_index.shape[1])
    if knn_metric == 'L2_normalized':
        faiss.normalize_L2(features_ref_index)
    knn_index.add(features_ref_index)
    time_memorybank = time.time() - start_time

    def mask_fn(features, grid_size):
        if masking:
            return model.compute_background_mask(features, grid_size, threshold=10,
                                                 masking_type=masking)
        return np.ones(features.shape[0], dtype=bool)

    inference_times = {}
    anomaly_scores = {}

    for type_anomaly in tqdm(type_anomalies, desc="Testing ({})".format(object_name)):
        data_dir = os.path.join(data_root, object_name, 'test', type_anomaly)
        gt_dir = os.path.join(data_root, object_name, 'ground_truth', type_anomaly)
        if save_patch_dists:
            os.makedirs(os.path.join(plots_dir, 'anomaly_maps', 'seed={}'.format(seed),
                                     object_name, 'test', type_anomaly), exist_ok=True)

        for idx, img_test_nr in enumerate(sorted(os.listdir(data_dir))):
            t0 = time.time()
            image_test = cv2.cvtColor(
                cv2.imread(os.path.join(data_dir, img_test_nr), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB)

            img_stem = os.path.splitext(img_test_nr)[0]
            if args.dataset == "VisA":
                gt_mask_path = os.path.join(gt_dir, '{}.png'.format(img_stem))
            else:
                gt_mask_path = os.path.join(gt_dir, '{}_mask.png'.format(img_stem))
            if os.path.exists(gt_mask_path):
                mask = model.prepare_mask(Image.open(gt_mask_path))
            else:
                H, W, C = image_test.shape
                mask = model.prepare_mask(np.zeros((H, W)))

            should_attack = (attack_type != 'none' and
                             (attack_target_samples == 'all' or
                              (attack_target_samples == 'anomaly' and type_anomaly != 'good') or
                              (attack_target_samples == 'normal' and type_anomaly == 'good')))
            # Apply attack
            if should_attack:
                if attack_type == 'fgsm':
                    with torch.enable_grad():
                        image_tensor, grid_size = fgsm_trained_probe(
                            model, probe, image_test, mask,
                            epsilon=epsilon, device=device)
                elif attack_type == 'pgd':
                    with torch.enable_grad():
                        image_tensor, grid_size = pgd_trained_probe(
                            model, probe, image_test, mask,
                            epsilon=epsilon, num_iter=pgd_iters,
                            device=device, random_start=True)
            else:
                image_tensor, grid_size = model.prepare_image(image_test)

            # Apply defense
            score, d_masked = apply_defenses_to_score(
            model, image_tensor, grid_size, knn_index,
            knn_metric=knn_metric, knn_neighbors=knn_neighbors,
            mask_fn=mask_fn if masking else None,
            patch_shift=patch_shift,
            feature_purifier=feature_purifier)

            torch.cuda.synchronize()
            key = '{}/{}'.format(type_anomaly, img_test_nr)
            inference_times[key] = time.time() - t0
            anomaly_scores[key] = score

            if save_tiffs:
                anomaly_map = dists2map(d_masked, image_test.shape)
                tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)

            if save_patch_dists:
                np.save(os.path.join(plots_dir, 'anomaly_maps', 'seed={}'.format(seed),
                                     object_name, 'test', type_anomaly,
                                     '{}.npy'.format(img_stem)), d_masked)

    return anomaly_scores, time_memorybank, inference_times


if __name__ == "__main__":
    args = parse_args()
    print("Defense: {}, Attack: {}, eps={:.5f}".format(
        args.defense, args.attack_type, args.epsilon))

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(
        args.dataset, args.preprocess, data_root=args.data_root)
    if args.objects is not None:
        invalid = [o for o in args.objects if o not in objects]
        if invalid:
            raise ValueError("Unknown objects: {}".format(invalid))
        objects = [o for o in objects if o in args.objects]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution)
    seeds = [args.just_seed] if args.just_seed is not None else range(args.num_seeds)

    patch_shift, feature_purifier = None, None

    if args.defense == 'patchshift':
        patch_shift = PatchShiftDefense(
            n_shifts=args.n_shifts, max_shift=args.max_shift,
            patch_size=args.patch_size, apply_spatial_filter=True,
            spatial_kernel=args.kernel_size)
    elif args.defense == 'purifier':
        feature_purifier = FeaturePurifier(
            lr=args.purifier_lr, n_epochs=args.purifier_epochs,
            noise_std=args.purifier_noise_std,
            n_noise_augments=args.purifier_n_noise, device='cuda',
            attack_type=args.purifier_attack,
            pgd_iters=args.purifier_pgd_iters)
    elif args.defense == 'patchshift_purifier':
        patch_shift = PatchShiftDefense(
            n_shifts=args.n_shifts, max_shift=args.max_shift,
            patch_size=args.patch_size, apply_spatial_filter=True,
            spatial_kernel=args.kernel_size)
        feature_purifier = FeaturePurifier(
            lr=args.purifier_lr, n_epochs=args.purifier_epochs,
            noise_std=args.purifier_noise_std,
            n_noise_augments=args.purifier_n_noise, device='cuda',
            attack_type=args.purifier_attack,
            pgd_iters=args.purifier_pgd_iters)
    
    attack_tag = args.attack_type
    defense_tag = args.defense
    if 'patchshift' in args.defense:
        defense_tag += '_n{}_s{}'.format(args.n_shifts, args.max_shift)
    if 'purifier' in args.defense:
        defense_tag += '_e{}_{}'.format(args.purifier_epochs, args.purifier_attack)
        if args.purifier_attack in ('pgd', 'adaptive_knn'):
            defense_tag += str(args.purifier_pgd_iters)
    if args.tag:
        defense_tag += '_' + args.tag

    results_dir = "results_{}/{}_{}/" \
                  "{}-shot_preprocess={}_attack={}_eps={:.5f}_defense={}".format(
                      args.dataset, args.model_name, args.resolution,
                      args.shots, args.preprocess, attack_tag,
                      args.epsilon, defense_tag)
    os.makedirs(results_dir, exist_ok=True)
    print("Results: {}".format(results_dir))

    with open(os.path.join(results_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    for seed in seeds:
        print("=========== Shot={}, Seed={} ===========".format(args.shots, seed))
        if os.path.exists(os.path.join(results_dir, 'metrics_seed={}.json'.format(seed))):
            print("Already exists, skipping.")
            continue

        timeit_file = os.path.join(results_dir, 'time_measurements.csv')
        with open(timeit_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Object", "Sample", "Anomaly_Score",
                             "MemoryBank_Time", "Inference_Time"])

            for obj in objects:
                print("\n--- {} ---".format(obj))
                for _ in trange(args.warmup_iters, desc="Warmup", leave=False):
                    first = os.listdir(os.path.join(args.data_root, obj, 'train', 'good'))[0]
                    t, g = model.prepare_image(
                        os.path.join(args.data_root, obj, 'train', 'good', first))
                    _ = model.extract_features(t)

                scores, t_bank, t_inf = run_defended_detection(
                    model, obj, args.data_root, args.shots, object_anomalies, results_dir,
                    attack_type=args.attack_type,
                    attack_target_samples=args.attack_target_samples,
                    epsilon=args.epsilon, pgd_iters=args.pgd_iters,
                    patch_shift=patch_shift, feature_purifier=feature_purifier,
                    masking=masking_default[obj], mask_ref_images=args.mask_ref_images,
                    rotation=rotation_default[obj], knn_metric=args.knn_metric,
                    knn_neighbors=args.k_neighbors, faiss_on_cpu=args.faiss_on_cpu,
                    seed=seed, save_patch_dists=True, save_tiffs=False, probe_n_train_images=args.probe_n_train_images, 
                    probe_epochs=args.probe_epochs, probe_hidden_dim=args.probe_hidden_dim)

                for sample in scores:
                    writer.writerow([obj, sample, "{:.5f}".format(scores[sample]),
                                     "{:.5f}".format(t_bank),
                                     "{:.5f}".format(t_inf[sample])])

        print("\n=========== Evaluate seed={} ===========".format(seed))
        eval_finished_run(args.dataset, args.data_root,
                          anomaly_maps_dir=os.path.join(
                              results_dir, 'anomaly_maps', 'seed={}'.format(seed)),
                          output_dir=results_dir, seed=seed,
                          eval_clf=True, eval_segm = False)

    print("\nDone!")