FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download NLTK data at build time (avoids cold-start delay)
RUN python -c "import nltk; nltk.download('words'); nltk.download('wordnet')"


COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]
