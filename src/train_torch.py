import argparse, os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class RatingDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, usecols=["userId", "movieId", "rating"])
        self.u2i = {u: i for i, u in enumerate(df["userId"].unique())}
        self.m2i = {m: i for i, m in enumerate(df["movieId"].unique())}
        self.users   = torch.tensor([self.u2i[u] for u in df["userId"]], dtype=torch.long)
        self.items   = torch.tensor([self.m2i[m] for m in df["movieId"]], dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,                dtype=torch.float32)

    def __len__(self):            return len(self.ratings)
    def __getitem__(self, idx):   return self.users[idx], self.items[idx], self.ratings[idx]


class MF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user = nn.Embedding(n_users, embed_dim)
        self.item = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user.weight, 0, 0.05)
        nn.init.normal_(self.item.weight, 0, 0.05)

    def forward(self, u, i):
        return (self.user(u) * self.item(i)).sum(1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("▶ device:", device)

    ds = RatingDataset(args.data)
    n_users   = int(ds.users.max()) + 1          # ← dùng cho checkpoint
    n_items   = int(ds.items.max()) + 1
    embed_dim = args.dim

    model = MF(n_users, n_items, embed_dim).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=2, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, ncols=100, desc=f"Epoch {epoch}/{args.epochs}")
        for u, i, r in pbar:
            u, i, r = u.to(device), i.to(device), r.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(u, i), r)
            loss.backward();  opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        ckpt_path = f"model/mf_epoch{epoch}.pt"
        os.makedirs("model", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "n_users":     n_users,
            "n_items":     n_items,
            "embedding_dim": embed_dim,
            "epoch":       epoch
        }, ckpt_path)
        print(f"💾  Saved {ckpt_path}")

    print("✅ Training finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim",    type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    main(ap.parse_args())
