# Use an official Python runtime with audio support
FROM python:3.11-slim

# Install system dependencies for audio and BentoML
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY twi_asr_service.py .
COPY bentofile.yaml .

# Expose the BentoML port
EXPOSE 8001

# Start the service
CMD ["bentoml", "serve", "twi_asr_service:TwiASRService", "--port", "8001"]
