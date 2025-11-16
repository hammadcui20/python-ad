# Dockerfile
FROM python:3.11-slim

# ffmpeg for media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV UPLOAD_DIR=/tmp/flask_ad_detector_uploads
EXPOSE 8000

CMD ["python", "app.py"]