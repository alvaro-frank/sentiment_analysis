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

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

EXPOSE 80

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]