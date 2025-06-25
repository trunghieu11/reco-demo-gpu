import os, json, torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from train_torch import MF   # tái sử dụng class

MODEL_PATH = os.getenv("MODEL_PATH", "model/mf_best.pt")
ckpt = torch.load(MODEL_PATH, map_location="cpu")
model = MF(ckpt["n_users"], ckpt["n_items"], ckpt["embedding_dim"])
model.load_state_dict(ckpt["model_state"])
model.eval()

app = FastAPI()

class Req(BaseModel):
    user_id: int
    k: int = 10

@torch.no_grad()
@app.post("/recommend")
def recommend(req: Req):
    if req.user_id >= ckpt["n_users"]:
        raise HTTPException(404, "user_id not found")

    user = torch.tensor([req.user_id])
    # dự đoán cho **toàn bộ** item vector  (GPU không cần thiết)
    scores = model.user(user) @ model.item.weight.T   # (1, n_items)
    _, topk = torch.topk(scores, req.k)
    return {"user_id": req.user_id,
            "recommendations": topk.squeeze(0).tolist()}
