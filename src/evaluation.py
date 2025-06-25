#!/usr/bin/env python3
"""Simple offline evaluation script for the MF model you just trained.

Metrics implemented
-------------------
* HR@K  – Hit‑Rate (a.k.a. Recall@K) per user then averaged
* NDCG@K – Normalised Discounted Cumulative Gain

Usage
-----
python evaluation.py \
    --ratings data/ml20/rating.csv \
    --model-path artifacts/model_latest.pt \
    --k 10 --test-frac 0.2

The script expects the **checkpoint** saved by ``export.py`` where
``ckpt['model_state']`` stores the PyTorch state‑dict and
``ckpt['meta']`` stores ``n_users`` and ``n_items``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

###############################################################################
# Model definition (same as in train_torch.py)                                #
###############################################################################

class MF(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.user = torch.nn.Embedding(n_users, emb_dim)
        self.item = torch.nn.Embedding(n_items, emb_dim)

    def forward(self, u, i):
        return (self.user(u) * self.item(i)).sum(1)

###############################################################################
# Utility functions                                                            #
###############################################################################

def dcg(rels: np.ndarray) -> float:
    """Discounted cumulative gain."""
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float((rels * discounts).sum())

def compute_metrics(ranked_lists: list[list[int]], ground_truth: list[set[int]], k: int) -> Tuple[float, float]:
    hits, ndcgs = [], []
    for recs, true_items in zip(ranked_lists, ground_truth):
        if not true_items:
            continue  # user had no test interaction
        topk = recs[:k]
        rels = np.array([1 if item in true_items else 0 for item in topk], dtype=np.float32)
        hits.append(rels.any())
        idcg = dcg(np.ones(min(len(true_items), k), dtype=np.float32))
        ndcgs.append(dcg(rels) / idcg if idcg > 0 else 0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))

###############################################################################
# Main evaluation routine                                                     #
###############################################################################

def main(args):
    ratings = pd.read_csv(args.ratings)
    # Map IDs to 0..N-1
    user2idx = {u: i for i, u in enumerate(ratings['userId'].unique())}
    item2idx = {v: i for i, v in enumerate(ratings['movieId'].unique())}
    ratings['u_idx'] = ratings['userId'].map(user2idx)
    ratings['i_idx'] = ratings['movieId'].map(item2idx)

    # Train/test split by timestamp (leave‑last X%% for test)
    ratings = ratings.sort_values('timestamp')
    cutoff = int(len(ratings) * (1 - args.test_frac))
    train_df, test_df = ratings.iloc[:cutoff], ratings.iloc[cutoff:]

    n_users, n_items = len(user2idx), len(item2idx)

    # Load model
    ckpt = torch.load(args.model_path, map_location='cpu')
    emb_dim = ckpt.get('meta', {}).get('emb_dim', 64)
    model = MF(n_users, n_items, emb_dim)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(args.device)

    # Build user -> items interactions dictionaries
    train_inter = train_df.groupby('u_idx')['i_idx'].apply(set).to_dict()
    test_inter = test_df.groupby('u_idx')['i_idx'].apply(set).to_dict()

    all_items = torch.arange(n_items, device=args.device)

    ranked_lists, ground_truth = [], []
    with torch.inference_mode():
        for u, test_items in tqdm(test_inter.items(), desc='Ranking users'):
            u_tensor = torch.tensor([u], device=args.device)
            # Exclude items seen in train
            seen = train_inter.get(u, set())
            mask = torch.ones(n_items, dtype=torch.bool, device=args.device)
            if seen:
                seen_idx = torch.tensor(list(seen), device=args.device)
                mask[seen_idx] = False
            candidates = all_items[mask]
            users_rep = u_tensor.repeat_interleave(len(candidates))
            scores = model(users_rep, candidates).cpu()
            topk_idx = torch.topk(scores, k=args.k)[1]
            rec_items = candidates[topk_idx].cpu().tolist()
            ranked_lists.append(rec_items)
            ground_truth.append(test_items)

    hr, ndcg = compute_metrics(ranked_lists, ground_truth, args.k)
    print(json.dumps({'HR@%d' % args.k: hr, 'NDCG@%d' % args.k: ndcg}, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratings', type=Path, required=True, help='ratings.csv path')
    ap.add_argument('--model-path', type=Path, required=True, help='checkpoint .pt file')
    ap.add_argument('--test-frac', type=float, default=0.2, help='fraction for test split')
    ap.add_argument('-k', '--k', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    main(args)

