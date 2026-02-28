# 🎙️ Twi (Akan) Speech Recognition Inference Server

![Hero Image](./hero_image.png)

---

## 🌟 The Vision: Bridging the Language Gap

Speech recognition for low-resource languages has long been a challenge, but through fine-tuning and the power of open-source models like Whisper, the barriers are breaking down. This repository houses the inference engine for the **`teckedd/whisper-small-serlabs-twi-asr`** model, specifically tuned to transcribe Twi (Akan), a widely spoken language in Ghana. 

We built this server with one major philosophy in mind: **Simplicity meets Scalability**. By wrapping our model inside a robust [BentoML](https://bentoml.com/) service, we've decoupled the complexity of PyTorch inference from the clean API surface developers need.

---

## 🏗️ The Architecture

At its core, this project leverages:
- **🧠 The Brain**: `Whisper-Small` model fine-tuned for Twi, leveraging the powerful `transformers` library.
- **🚀 The Engine**: BentoML runtime, abstracting away batching, concurrency, and API hosting into an elegant production service.
- **🎨 The Interface**: An interactive, out-of-the-box Gradio UI that allows users to record and transcribe audio using a stunning web interface. 

---

## 🎮 Hands-On: The Interactive Web UI

No need to wait—let's interact with the model right away! We've included a sleek, beautiful **Gradio Web UI** that you can spin up alongside the server.

Once the API Server is running, simply execute:

```bash
uv run python ui.py
```

This will launch a local web application at `http://127.0.0.1:7860`. Open your browser to:
1. Speak natively into your microphone in Twi.
2. Upload a pre-recorded `.wav` file.
3. Click **Transcribe Audio ✨** and watch the magic happen!

---

## ⚡ Quickstart Guide

Let's get the server running on your local machine.

### Option 1: Docker (Recommended)

Thanks to the provided `Dockerfile`, deploying the server is just two commands away.

Build the image:
```bash
docker build -t twi-asr-server .
```

Run the container:
```bash
docker run -p 8001:8001 twi-asr-server
```
*Your server is now alive and transcribing at `http://localhost:8001/docs`!*

### Option 2: Local Python Environment (`uv` recommended)

Prefer to run it natively without Docker? No problem! We highly recommend using `uv` for lightning-fast virtual environment installation.

1. **Install System Dependencies** (Debian/Ubuntu):
   ```bash
   sudo apt-get install libsndfile1 ffmpeg curl
   ```

2. **Initialize Environment & Install Python Libraries**:
   ```bash
   # Initialize uv venv and activate
   uv venv
   source .venv/bin/activate
   
   # Install dependencies rapidly
   uv pip install -r requirements.txt
   ```

3. **Ignite the Server**:
   ```bash
   uv run bentoml serve twi_asr_service:TwiASRService --port 8001
   ```

---

## 🧩 API Reference

If you are a developer looking to integrate this into your application, the server provides a clean REST API. 

The Swagger UI is fully interactive and available at `http://localhost:8001/docs`.

### 1. The Transcription Endpoint
- **URL**: `POST /transcribe`
- **Payload**: Form-data with `audio_file` as an audio blob/file.
- **Returns**: JSON object containing the `transcript` and processing `metadata`.

**Test via cURL**:
```bash
curl -X POST http://localhost:8001/transcribe \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@my_audio.wav"
```

### 2. The Health Check Endpoint
- **URL**: `GET /health`
- **Returns**: Details about the server status, device usage (`cpu` or `cuda`), and PyTorch version.

**Test via cURL**:
```bash
curl http://localhost:8001/health
```

---

## 💡 What's Happening Under The Hood?

When an audio file hits the `/transcribe` endpoint, here is the journey it takes:

1. **Intake**: BentoML accepts the raw bytes, routing it with high concurrency configurations.
2. **Preprocessing**: The audio is loaded using `librosa` and resampled to the model's expected 16kHz standard.
3. **Inference**: The file passes through the Whisper feature extractor and enters the PyTorch model running optimally (in mixed-precision FP16 if a GPU is detected!).
4. **Delivery**: Raw IDs are decoded back into a readable Twi transcript, returning almost instantly back to the client.

Enjoy the project and happy transcribing! 🚀