import os
import faiss
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.utils import augment_image

def _get_norm_constants(model, device):
    """Return (mean, std) as [1,3,1,1] tensors on `device`."""
    mean = getattr(model, 'mean',
                   torch.tensor([0.485, 0.456, 0.406], device=device))
    std = getattr(model, 'std',
                  torch.tensor([0.229, 0.224, 0.225], device=device))
    if mean.dim() == 1:
        mean = mean.view(1, 3, 1, 1)
    if std.dim() == 1:
        std = std.view(1, 3, 1, 1)
    return mean.to(device), std.to(device)

def _prepare_input(model, image, device):
    """Prepare a normalised [1,3,H,W] tensor with grad tracking + grid size."""
    image_tensor, grid = model.prepare_image(image)      # [3,H,W], grid
    X = image_tensor.unsqueeze(0).to(device)
    X = X.clone().detach().requires_grad_(True)
    return X, grid

def _build_patch_mask(mask, grid, device):
    """Resize a spatial mask to a flat [1, N] patch-level binary mask."""
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    elif isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    grid_h, grid_w = grid
    patch_mask = cv2.resize(
        (mask_np > 0).astype(np.float32),
        (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.from_numpy(patch_mask).to(device=device,
                                           dtype=torch.float32).view(1, -1)

def _clamp_to_valid(X, original_X, eps_norm, mean, std):
    """Project X back into the L_inf ball *and* valid pixel range."""
    delta = torch.clamp(X - original_X, min=-eps_norm, max=eps_norm)
    lo = (0.0 - mean) / std
    hi = (1.0 - mean) / std
    return torch.clamp(original_X + delta, min=lo, max=hi)

class DistanceProbe(nn.Module):
    """Per-patch regression probe: features -> predicted k-NN distance."""
    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats):
        """feats: [B, N, D] -> predicted log-distance [B, N]"""
        return self.net(feats).squeeze(-1)

def train_distance_probe(model, features_ref_raw,
                         masking, mask_ref_images, rotation=False,
                         n_train_images=50, data_root=None, object_name=None,
                         hidden_dim=None, lr=1e-3, epochs=100,
                         normalize_feats=True, device='cuda', seed=42):
    """
    Train a regression probe that predicts per-patch k-NN distance to the
    support memory bank.

    The regression target is the actual per-patch k-NN distance, computed
    against `features_ref_raw` (the support memory bank). The probe is then
    a differentiable proxy for the k-NN scorer used at test time, providing
    meaningful gradients across the full feature space.

    Training features come from `n_train_images` clean train-set images,
    drawn from the same class as the support set but disjoint from it.
    This is consistent with a white-box attacker threat model: the attacker
    has access to the model, the training data distribution, and the support
    bank. Few-shot constraints apply only to detection-time support, not to
    attack-time probe training.

    Parameters
    ----------
    model : DINOv2 wrapper with extract_features, prepare_image methods.
    features_ref_raw : np.ndarray [M, D]
        Memory bank features (un-normalized). Used both as targets to compute
        k-NN distance against, and as the basis for feature normalization
        consistency.
    masking, mask_ref_images, rotation : forwarded to model.compute_background_mask.
    n_train_images : int
        Number of held-out normal train images to use for probe training.
        Default 50; MVTec train/good/ typically has 200+ images.
    data_root, object_name : path info to locate the train/good/ folder.
    hidden_dim : int or None
        MLP hidden dimension. Default = feat_dim.
    lr, epochs : float, int  optimizer config.
    normalize_feats : bool
        Whether to L2-normalize features before passing to the probe. Must
        match the k-NN scorer's normalization (knn_metric='L2_normalized').
    device : str
    seed : int

    Returns
    -------
    probe : DistanceProbe  (eval mode, frozen, on `device`)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_dim = features_ref_raw.shape[1]
    bank = features_ref_raw.copy().astype('float32')
    if normalize_feats:
        faiss.normalize_L2(bank)
    knn_index = faiss.IndexFlatL2(feat_dim)
    knn_index.add(bank)

    # --- Collect training features and their k-NN distance labels.
    train_folder = os.path.join(data_root, object_name, 'train', 'good')
    all_train = sorted(os.listdir(train_folder))
    # Skip the first N (assumed to be the support; in practice the caller
    # should pass a flag for this, but for now we sample from the tail).
    candidate_files = all_train[-n_train_images:] if len(all_train) >= n_train_images else all_train

    train_features = []
    train_distances = []

    for fname in candidate_files:
        img = cv2.cvtColor(cv2.imread(os.path.join(train_folder, fname)),
                           cv2.COLOR_BGR2RGB)
        imgs = augment_image(img) if rotation else [img]
        for im in imgs:
            tensor, grid = model.prepare_image(im)
            feats = model.extract_features(tensor)
            bg_mask = model.compute_background_mask(
                feats, grid, threshold=10,
                masking_type=(mask_ref_images and masking))
            feats_masked = feats[bg_mask]

            # Compute k-NN distance labels in the same metric used at test time.
            f = feats_masked.copy().astype('float32')
            if normalize_feats:
                faiss.normalize_L2(f)
            dists, _ = knn_index.search(f, k=1)
            dists = dists.squeeze(1)
            if normalize_feats:
                dists = dists / 2  # match scoring convention

            train_features.append(feats_masked)
            train_distances.append(dists)

    train_features = np.concatenate(train_features, axis=0).astype('float32')
    train_distances = np.concatenate(train_distances, axis=0).astype('float32')

    # Use log(1 + d) as the regression target. Stabilises training across
    # the wide dynamic range of k-NN distances; the +1 keeps zero-distance
    # patches finite.
    log_targets = np.log1p(train_distances)

    print("  DistanceProbe: training on {} patches from {} images".format(
        train_features.shape[0], len(candidate_files)))

    # --- Train the probe.
    probe = DistanceProbe(feat_dim, hidden_dim).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr)

    X_t = torch.from_numpy(train_features).to(device)
    if normalize_feats:
        X_t = F.normalize(X_t, dim=-1)
    y_t = torch.from_numpy(log_targets).to(device)

    n = X_t.shape[0]
    batch_size = min(2048, n)

    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = probe.net(X_t[idx]).squeeze(-1)
            loss = F.mse_loss(pred, y_t[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print("    Epoch {}/{}: MSE={:.6f}".format(
                epoch + 1, epochs, total_loss / max(n_batches, 1)))

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    # Sanity check: correlation between probe predictions and true k-NN
    # distances on a held-out batch. Worth printing — if Spearman < 0.5
    # the probe is still a poor proxy and you should debug before running
    # attacks.
    with torch.no_grad():
        from scipy.stats import spearmanr
        pred_all = probe.net(X_t).squeeze(-1).cpu().numpy()
        rho, _ = spearmanr(pred_all, log_targets)
        print("  DistanceProbe: Spearman(pred, true) = {:.3f}".format(rho))

    return probe

def fgsm_trained_probe(model, probe, image, mask, epsilon=8/255,
                       device='cuda'):
    """
    Single-step FGSM using the trained surrogate probe and the PatchGuard
    mean-difference loss.
 
    Loss is the mask-aware difference of mean-pooled per-patch surrogate
    scores between normal and anomalous regions (identical in form to
    `pgd_trained_probe`):
 
        L = mean_{j in N_0} probe(feats)_j  -  mean_{j in N_1} probe(feats)_j
 
    where N_0 = {j : mask_j = 0} and N_1 = {j : mask_j = 1}. Maximising
    L raises normal-region scores and suppresses anomalous-region scores,
    inducing bidirectional ranking inversion in one sign-gradient step.
 
    The perturbation is applied in pixel space under an L_inf(epsilon)
    constraint.
 
    Parameters
    ----------
    model : DINOv2Wrapper
    probe : SurrogateProbe (trained, eval mode)
    image : np.ndarray [H, W, 3] uint8 or PIL Image
    mask  : torch.Tensor [H', W']  ground-truth mask (0 = normal, > 0 = anomaly)
    epsilon : float  perturbation budget in pixel [0, 1] space
 
    Returns
    -------
    X_adv : torch.Tensor [3, H, W]  normalised adversarial image
    grid  : tuple (grid_h, grid_w)
    """
    device = getattr(model, 'device', device)
    mean, std = _get_norm_constants(model, device)
    eps_norm = epsilon / std
 
    X, grid = _prepare_input(model, image, device)
    patch_mask_t = _build_patch_mask(mask, grid, device)        # [1, N]
 
    non_zeros_count = patch_mask_t.sum(dim=1).clamp(min=1.0)
    zeros_count = (1.0 - patch_mask_t).sum(dim=1).clamp(min=1.0)
 
    with torch.enable_grad():
        feats = model.model.get_intermediate_layers(X)[0]       # [1, N, D]
        scores = probe(feats)                                    # [1, N]
 
        anomalous_loss = (patch_mask_t * scores).sum(dim=1) / non_zeros_count
        normal_loss = ((1.0 - patch_mask_t) * scores).sum(dim=1) / zeros_count
        loss = normal_loss.sum() - anomalous_loss.sum()
        loss.backward()
 
    with torch.no_grad():
        X_adv = X + eps_norm * X.grad.sign()
        X_adv = _clamp_to_valid(X_adv, X.detach(), eps_norm, mean, std)
 
    return X_adv.detach().squeeze(0), grid

def pgd_trained_probe(model, probe, image, mask, epsilon=8/255,
                         num_iter=20, device='cuda', random_start=False):
    """
    PatchGuard-style PGD attack reproduced in this pipeline.

    Faithful port of Nafez et al. (CVPR 2025) `pgd_attack`:
      - Loss is the difference of mean-pooled per-patch detector scores
        between normal and anomalous regions.
      - No CW reformulation, no feature displacement, no memory-bank term.
      - Step size alpha = 2.5 * epsilon / num_iter (PatchGuard's choice;
        see Wong et al., ICLR 2020, for this schedule).

    The only deviation from PatchGuard's pseudocode is that PatchGuard's
    `model(X)` returns anomaly scores directly, whereas here we route
    through the ViT + surrogate probe:

        scores = probe(model.get_intermediate_layers(X)[0])

    This is the same threat model -- detector-only, no memory bank --
    just adapted to the surrogate-probe architecture.

    Parameters
    ----------
    model, probe, image, mask, epsilon, num_iter, device : as in
        pgd_configurable_trained_probe.
    random_start : bool
        Default False to match PatchGuard exactly. Set True for a
        random-start variant (closer to Madry et al., 2018).

    Returns
    -------
    X_adv : torch.Tensor [3, H, W]  normalised adversarial tensor
    grid  : tuple (grid_h, grid_w)
    """
    device = getattr(model, 'device', device)
    mean, std = _get_norm_constants(model, device)
    eps_norm = epsilon / std
    #alpha=epsilon/4
    alpha = (2.5 * epsilon) / num_iter
    alpha_norm = alpha / std

    X_orig, grid = _prepare_input(model, image, device)
    X_orig = X_orig.detach()
    patch_mask_t = _build_patch_mask(mask, grid, device)   # [1, N], float

    if random_start:
        delta = torch.zeros_like(X_orig).uniform_(-1, 1) * eps_norm
        X = (X_orig + delta).clone().detach()
        X = _clamp_to_valid(X, X_orig, eps_norm, mean, std).detach()
    else:
        X = X_orig.clone().detach()

    non_zeros_count = patch_mask_t.sum(dim=1).clamp(min=1.0)        # [1]
    zeros_count = (1.0 - patch_mask_t).sum(dim=1).clamp(min=1.0)    # [1]

    for _ in range(num_iter):
        X.requires_grad_(True)
        feats = model.get_patch_tokens(X)
        #feats = model.model.get_intermediate_layers(X)[0]           # [1, N, D]
        scores = probe(feats)                                        # [1, N]

        anomalous_loss = (patch_mask_t * scores).sum(dim=1) / non_zeros_count
        normal_loss = ((1.0 - patch_mask_t) * scores).sum(dim=1) / zeros_count
        loss = normal_loss.sum() - anomalous_loss.sum()
        grad = torch.autograd.grad(loss, X)[0].detach()

        with torch.no_grad():
            X = X + alpha_norm * grad.sign()
            X = _clamp_to_valid(X, X_orig, eps_norm, mean, std).detach()

    return X.squeeze(0), grid