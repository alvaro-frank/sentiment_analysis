FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=10000 --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt wordnet punkt_tab

COPY . .

CMD ["python", "src/train.py"]