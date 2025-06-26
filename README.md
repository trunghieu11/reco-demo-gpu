# Recommender Demo (MovieLens 20M)

## 📁 Folder layout

```
.
├── data/                 # 👉 nơi chứa dataset (không commit lên Git)
│   └── raw/              #   └─ rating.csv, movies.csv …
├── models/               # 👉 nơi lưu checkpoint .pt (không commit)
├── build_ctx/            # Docker build context
├── src/                  # code chính (train_torch.py, serve.py …)
└── README.md             # tài liệu này
```

---

## 1️⃣ Tải MovieLens 20M từ Kaggle

> **Yêu cầu**: có tài khoản Kaggle & đã tạo API token (`kaggle.json`).

```bash
# 0. Cài CLI
pip install -U kaggle

# 1. Chép token
mkdir -p ~/.kaggle
cp <đường_dẫn>/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Tạo thư mục dữ liệu trong project
cd /path/to/reco-demo-gpu
mkdir -p data/raw

# 3. Download + unzip
kaggle datasets download -d grouplens/movielens-20m-dataset -p data/raw
unzip data/raw/movielens-20m-dataset.zip -d data/raw
rm data/raw/*.zip
```

Kết quả bạn cần là file `data/raw/rating.csv` (\~1,7 GB).

---

## 2️⃣ Train + Eval cục bộ (GPU)

```bash
# 1. Tạo virtualenv (nếu chưa)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # đã có tqdm, pandas, torch…

# 2. Train
python src/train_torch.py \
  --data data/raw/rating.csv \
  --batch 64 --epochs 5 --dim 64

# 3. Đánh giá
python src/evaluation.py \
  --ratings data/raw/rating.csv \
  --model-path models/mf_epoch5.pt \
  --k 10
```

`src/train_torch.py` sẽ tự lưu checkpoint trong `models/`; file tốt nhất bạn chọn (ví dụ `mf_best.pt`) để build.

---

## 3️⃣ Đóng gói Docker & đẩy lên Artifact Registry

```bash
# 0. (Một lần duy nhất) – tạo repo docker trong Artifact Registry:
gcloud artifacts repositories create reco-demo \
    --repository-format=docker \
    --location=asia-southeast1

# 1. Copy model vào build_ctx
cp models/mf_best.pt build_ctx/models/

# 2. Tag & build
PROJECT_ID=$(gcloud config get-value project)
REGION=asia-southeast1
REPO=${REGION}-docker.pkg.dev/$PROJECT_ID/reco-demo/reco
TAG=$(git rev-parse --short HEAD)
IMAGE="$REPO:$TAG"

gcloud builds submit build_ctx --tag "$IMAGE"
```

> **Dockerfile** trong `build_ctx/` đã cài đủ:\
> `pandas`, `tqdm`, `uvicorn`, `fastapi`, `torch==2.3`, v.v.

---

## 4️⃣ Triển khai Cloud Run

```bash
gcloud run deploy reco-v2 \
  --image "$IMAGE" \
  --region asia-southeast1 \
  --memory 2Gi --cpu 2 \
  --timeout 600 \
  --allow-unauthenticated
```

Sau khi “Service URL” xuất hiện, kiểm tra:

```bash
curl -X POST \
  <SERVICE_URL>/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 42, "k": 10}'
```
