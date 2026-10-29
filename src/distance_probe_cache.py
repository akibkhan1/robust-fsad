"""
Cached wrapper around `train_distance_probe`.

The DistanceProbe depends only on (backbone, resolution, object, support
indices, preprocessing flags, probe hyperparameters). None of these
change across attack types or defenses. Re-training the probe on every
`run_defense.py` invocation is wasted work.

This module provides `train_or_load_distance_probe`, a drop-in
replacement that:
  - Computes a deterministic cache key from the relevant inputs.
  - If a cached probe exists with that key, loads and returns it.
  - Otherwise trains via `train_distance_probe`, saves the result, and
    returns it.

Cache layout
------------
<cache_root>/
  <key>.pt        # state_dict + metadata
  <key>.meta.yaml # human-readable record of what produced this probe

The key is a short hash of (backbone, resolution, object, shots, seed,
masking, mask_ref_images, rotation, n_train_images, epochs, hidden_dim,
lr, probe_seed). Hyperparameter changes invalidate the cache
automatically; no manual cache busting needed.
"""

import hashlib
import json
import os
import torch
from src.adversarial import DistanceProbe, train_distance_probe

def _cache_key(spec):
    """
    Deterministic short hash of all probe-determining parameters.

    `spec` is a dict of primitive values; we json-serialise with sorted
    keys so the hash is stable across Python sessions.
    """
    blob = json.dumps(spec, sort_keys=True).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()[:16]


def _cache_paths(cache_root, key):
    weights_path = os.path.join(cache_root, '{}.pt'.format(key))
    meta_path = os.path.join(cache_root, '{}.meta.json'.format(key))
    return weights_path, meta_path

def train_or_load_distance_probe(
    model, features_ref_raw,
    masking, mask_ref_images, rotation=False,
    n_train_images=50, data_root=None, object_name=None,
    hidden_dim=None, lr=1e-3, epochs=100,
    normalize_feats=True, device='cuda', seed=42,
    # ---- cache control ----
    backbone_name=None, resolution=None,
    shots=None, support_seed=None,
    cache_root='probe_cache',
    force_retrain=False,
    verbose=True,
):
    """
    Train a new DistanceProbe or load a cached one.

    All arguments before the `cache control` block are forwarded to
    `train_distance_probe` unchanged; behaviour matches that function
    when no cache hit occurs. Cache-determining metadata (the new args)
    must uniquely identify the (object, support set, backbone) tuple.

    Parameters
    ----------
    backbone_name : str
        e.g. 'dinov2_vits14'. Required for the cache key.
    resolution : int
        Input resolution (e.g. 448). Required for the cache key.
    shots : int
        Number of support images (k). Required for the cache key.
    support_seed : int
        The seed run_defense.py uses to slice the support images out
        of the train/good folder. Required for the cache key.
    cache_root : str
        Directory to read from / write to. Created if missing.
    force_retrain : bool
        If True, ignore any cached probe and retrain. Useful when the
        training code has changed in a way the cache key cannot
        capture.
    verbose : bool
        Print cache hit / miss info.

    Returns
    -------
    probe : DistanceProbe in eval mode on `device`, parameters frozen.
    """
    # All probe-determining variables go into the cache key. If any of
    # these changes, the cached probe is stale.
    required = {
        'backbone_name':  backbone_name,
        'resolution':     resolution,
        'object_name':    object_name,
        'shots':          shots,
        'support_seed':   support_seed,
        'masking':        bool(masking) if masking is not None else False,
        'mask_ref_images': bool(mask_ref_images),
        'rotation':       bool(rotation),
        'n_train_images': int(n_train_images),
        'epochs':         int(epochs),
        'hidden_dim':     hidden_dim,
        'lr':             float(lr),
        'normalize_feats': bool(normalize_feats),
        'probe_seed':     int(seed),
    }
    missing = [k for k, v in required.items() if v is None and k != 'hidden_dim']
    if missing:
        raise ValueError(
            "train_or_load_distance_probe: cache key requires {}; "
            "got None for these.".format(missing))

    key = _cache_key(required)
    os.makedirs(cache_root, exist_ok=True)
    weights_path, meta_path = _cache_paths(cache_root, key)

    # ---- Cache hit ----
    if (not force_retrain) and os.path.exists(weights_path):
        if verbose:
            print("  DistanceProbe: cache HIT  [{}/{}]  loading from {}".format(
                object_name, key, weights_path))
        # Reconstruct the architecture from the saved metadata so we use
        # the exact feat_dim / hidden_dim that produced the weights.
        with open(meta_path) as f:
            meta = json.load(f)
        feat_dim = meta['feat_dim']
        saved_hidden = meta.get('hidden_dim', None)
        probe = DistanceProbe(feat_dim, saved_hidden).to(device)
        probe.load_state_dict(torch.load(weights_path, map_location=device))
        probe.eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        return probe

    # ---- Cache miss: train ----
    if verbose:
        print("  DistanceProbe: cache MISS [{}/{}]  training".format(
            object_name, key))

    probe = train_distance_probe(
        model=model,
        features_ref_raw=features_ref_raw,
        masking=masking,
        mask_ref_images=mask_ref_images,
        rotation=rotation,
        n_train_images=n_train_images,
        data_root=data_root,
        object_name=object_name,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        normalize_feats=normalize_feats,
        device=device,
        seed=seed,
    )

    # ---- Save to cache ----
    torch.save(probe.state_dict(), weights_path)
    feat_dim = features_ref_raw.shape[1]
    meta = dict(required)
    meta['feat_dim'] = int(feat_dim)
    meta['cache_key'] = key
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    if verbose:
        print("  DistanceProbe: saved to {}".format(weights_path))

    return probe
