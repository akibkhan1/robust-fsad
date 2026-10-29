import numpy as np
import torch
import torch.nn as nn
import faiss
from scipy.ndimage import median_filter
from src.post_eval import mean_top1p
from src.utils import augment_image
from src.adversarial import pgd_trained_probe, fgsm_trained_probe

class PatchShiftDefense:
    """
    Random pixel shifts to misalign adversarial perturbation with
    the ViT's 14x14 patch grid.  Scores N shifted versions and
    takes the element-wise median of the distance maps.
    """

    def __init__(self, n_shifts=8, max_shift=7, patch_size=16,
                 apply_spatial_filter=True, spatial_kernel=3):
        self.n_shifts = n_shifts
        self.max_shift = max_shift
        self.patch_size = patch_size
        self.apply_spatial_filter = apply_spatial_filter
        self.spatial_kernel = spatial_kernel

    def score_image(self, model, image_tensor, grid_size, knn_index,
                    knn_metric='L2_normalized', knn_neighbors=1,
                    mask_fn=None, feature_purifier=None):
        img_t = image_tensor.detach().cpu()
        while img_t.dim() > 3:
            img_t = img_t.squeeze(0)
        C, H, W = img_t.shape
        scores = []
        all_grid_distances = []

        offsets = [(0, 0)]
        for _ in range(self.n_shifts - 1):
            dy = np.random.randint(0, self.max_shift + 1)
            dx = np.random.randint(0, self.max_shift + 1)
            offsets.append((dy, dx))

        for dy, dx in offsets:
            cropped = img_t[:, dy:, dx:]
            _, ch, cw = cropped.shape
            new_h = ch - ch % self.patch_size
            new_w = cw - cw % self.patch_size
            cropped = cropped[:, :new_h, :new_w]
            new_grid = (new_h // self.patch_size, new_w // self.patch_size)

            features = model.extract_features(cropped)

            if feature_purifier is not None:
                features = feature_purifier.purify(features)

            bg_mask = mask_fn(features, new_grid) if mask_fn else np.ones(features.shape[0], dtype=bool)
            features_masked = features[bg_mask]

            fc = features_masked.copy().astype('float32')
            if knn_metric == 'L2_normalized':
                faiss.normalize_L2(fc)
            distances, _ = knn_index.search(fc, k=knn_neighbors)
            if knn_neighbors > 1:
                distances = distances.mean(axis=1)
            if knn_metric == 'L2_normalized':
                distances = distances / 2

            od = np.zeros_like(bg_mask, dtype=float)
            od[bg_mask] = distances.squeeze()
            scores.append(mean_top1p(od.flatten()))
            n_tokens = features.shape[0]
            side = int(round(n_tokens ** 0.5))
            if side * side == n_tokens:
                new_grid = (side, side)
            #d_map = od.reshape(new_grid)
            d_map = od.reshape(new_grid)
            if new_grid != grid_size:
                import cv2
                d_map = cv2.resize(d_map.astype(np.float32),
                                   (grid_size[1], grid_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
            all_grid_distances.append(d_map)

        stacked = np.stack(all_grid_distances, axis=0)
        d_masked = np.median(stacked, axis=0)
        score = mean_top1p(d_masked.flatten())

        if self.apply_spatial_filter:
            d_masked = median_filter(d_masked, size=self.spatial_kernel)
            score = mean_top1p(d_masked.flatten())

        return score, d_masked


class PurifierMLP(nn.Module):
    """Lightweight residual MLP: adversarial features -> clean features."""

    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return x + self.net(x)


class FeaturePurifier:
    """
    Train a lightweight MLP on the k support images to map adversarial
    DINOv2 features back toward clean features.

    Adversarial training pairs are generated using one of three attacks,
    selected via attack_type:
      - 'fgsm'         : single-step probe-based FGSM
      - 'pgd'          : multi-step probe-based PGD

    """

    def __init__(self, feat_dim=384, hidden_dim=None,
                 lr=1e-3, n_epochs=100, noise_std=0.03,
                 n_noise_augments=4, device='cuda',
                 attack_type='fgsm', pgd_iters=10):
        if attack_type not in ('fgsm', 'pgd'):
            raise ValueError(
                "attack_type must be 'fgsm', or 'pgd',"
                "got '{}'".format(attack_type))
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.noise_std = noise_std
        self.n_noise_augments = n_noise_augments
        self.device = device
        self.attack_type = attack_type
        self.pgd_iters = pgd_iters
        self.model = None

    def train_purifier(self, model, support_images, masking, mask_ref_images,
                       rotation=False, probe=None, epsilon=8/255):
        """
        Train the feature purifier on the k support images.

        Parameters
        ----------
        model : DINOv2Wrapper
        support_images : list of np.ndarray [H,W,3] uint8
        masking, mask_ref_images : masking config
        rotation : bool
        probe : trained surrogate probe (required for fgsm/pgd attacks;
                ignored for adaptive_knn)
        epsilon : attack epsilon
        """

        clean_features = []
        adv_features = []

        for image in support_images:
            imgs = augment_image(image) if rotation else [image]
            for img in imgs:
                # --- Clean features ---
                tensor, grid = model.prepare_image(img)
                feats_clean = model.extract_features(tensor)
                bg_mask = model.compute_background_mask(
                    feats_clean, grid, threshold=10,
                    masking_type=(mask_ref_images and masking))
                feats_clean_masked = feats_clean[bg_mask]

                # --- Adversarial features ---
                # All support images are normal (mask all zeros); attack is
                # therefore in the "raise score" direction for fgsm/pgd alike.
                H, W = img.shape[:2]
                dummy_mask = model.prepare_mask(np.zeros((H, W)))
                with torch.enable_grad():
                    if self.attack_type == 'pgd':
                        if probe is None:
                            raise ValueError(
                                "attack_type='pgd' requires a trained probe.")
                        
                        adv_tensor, adv_grid = pgd_trained_probe(
                            model, probe, img, dummy_mask,
                            epsilon=epsilon, num_iter=self.pgd_iters,
                            device=model.device, random_start=True)

                    elif self.attack_type == 'fgsm':
                        if probe is None:
                            raise ValueError(
                                "attack_type='fgsm' requires a trained probe.")
                        
                        adv_tensor, adv_grid = fgsm_trained_probe(
                            model, probe, img, dummy_mask,
                            epsilon=epsilon, device=model.device)
                feats_adv = model.extract_features(adv_tensor)
                feats_adv_masked = feats_adv[bg_mask]

                clean_features.append(feats_clean_masked)
                adv_features.append(feats_adv_masked)

                # --- Noise-augmented pseudo-adversarial features ---
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                for _ in range(self.n_noise_augments):
                    noised_tensor = tensor.detach().cpu().clone()
                    noise = torch.randn_like(noised_tensor) * (self.noise_std / std)
                    noised_tensor = noised_tensor + noise
                    lo = (0.0 - mean) / std
                    hi = (1.0 - mean) / std
                    noised_tensor = torch.clamp(noised_tensor, lo, hi)

                    feats_noised = model.extract_features(noised_tensor)
                    feats_noised_masked = feats_noised[bg_mask]

                    clean_features.append(feats_clean_masked)
                    adv_features.append(feats_noised_masked)

        clean_features = np.concatenate(clean_features, axis=0).astype('float32')
        adv_features = np.concatenate(adv_features, axis=0).astype('float32')

        print("  FeaturePurifier: training on {} patch pairs (attack={})".format(
            clean_features.shape[0], self.attack_type))

        # --- Train the MLP ---
        self.model = PurifierMLP(self.feat_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        clean_t = torch.from_numpy(clean_features).to(self.device)
        adv_t = torch.from_numpy(adv_features).to(self.device)

        n = clean_t.shape[0]
        batch_size = min(2048, n)

        self.model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]

                purified_adv = self.model(adv_t[idx])
                loss_adv = nn.functional.mse_loss(purified_adv, clean_t[idx])

                purified_clean = self.model(clean_t[idx])
                loss_clean = nn.functional.mse_loss(purified_clean, clean_t[idx])

                loss = loss_adv + loss_clean

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 25 == 0 or epoch == 0:
                print("    Epoch {}/{}: loss={:.6f}".format(
                    epoch + 1, self.n_epochs, total_loss / max(n_batches, 1)))

        self.model.eval()
        print("  FeaturePurifier: trained")

    def purify(self, features):
        if self.model is None:
            return features
        with torch.inference_mode():
            feat_t = torch.from_numpy(features.astype('float32')).to(self.device)
            purified = self.model(feat_t)
        return purified.cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Combined defense pipeline
# ---------------------------------------------------------------------------

def apply_defenses_to_score(model, image_tensor, grid_size, knn_index,
                            knn_metric='L2_normalized', knn_neighbors=1,
                            mask_fn=None,
                            patch_shift=None,
                            feature_purifier=None):
    if patch_shift is not None:
        score, d_masked = patch_shift.score_image(
            model, image_tensor, grid_size, knn_index,
            knn_metric, knn_neighbors, mask_fn,
            feature_purifier=feature_purifier)
    else:
        features = model.extract_features(image_tensor)

        if feature_purifier is not None:
            features = feature_purifier.purify(features)

        bg_mask = mask_fn(features, grid_size) if mask_fn else np.ones(features.shape[0], dtype=bool)
        features = features[bg_mask]
        fc = features.copy().astype('float32')
        if knn_metric == 'L2_normalized':
            faiss.normalize_L2(fc)
        distances, _ = knn_index.search(fc, k=knn_neighbors)
        if knn_neighbors > 1:
            distances = distances.mean(axis=1)
        if knn_metric == 'L2_normalized':
            distances = distances / 2
        od = np.zeros_like(bg_mask, dtype=float)
        od[bg_mask] = distances.squeeze()
        d_masked = od.reshape(grid_size)
        score = mean_top1p(od.flatten())

    return score, d_masked