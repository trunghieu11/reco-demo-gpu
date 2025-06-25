FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/models/mf_best.pt
EXPOSE 8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
