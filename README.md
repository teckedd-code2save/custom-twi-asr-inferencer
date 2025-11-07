# Standalone Inference Server

## Solution: BentoML (Simplest)

### Project Structure

```
inference-server/
├── bentofile.yaml
├── twi_asr_service.py
├── requirements.txt
└── README.md
```

### Step 1: Install

```bas# Twi ASR Inference Server

Standalone inference server for Twi (Akan) speech recognition using Whisper model.

## Quick Start

### Step 1: Install Dependencies

```bash
pip install bentoml transformers librosa torch
```

### Step 2: Create `requirements.txt`

```txt
transformers==4.35.0
librosa==0.10.0
torch==2.0.0
numpy==1.24.3
```

### Step 3: Create `twi_asr_service.py`

```python
import bentoml
from bentoml.io import File, JSON
import torch
import librosa
import tempfile
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Define the model runner
class TwiASRModel(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        self.model.eval()
        print("✓ Model loaded")

    @bentoml.Runnable.method
    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcribe audio bytes and return transcript"""
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            # Load audio
            speech, sr = librosa.load(temp_path, sr=16000)
            
            # Transcribe
            with torch.no_grad():
                input_features = self.processor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcript = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return {"transcript": transcript}
        
        finally:
            os.remove(temp_path)

# Create runner
runner = bentoml.Runner(TwiASRModel, name="twi_asr_runner")

# Create service
svc = bentoml.Service(
    name="twi_asr_inference",
    runners=[runner],
)

# Define API endpoints
@svc.api(
    input=File(),
    output=JSON()
)
def transcribe_audio(audio_file) -> dict:
    """Transcribe uploaded audio file"""
    result = runner.transcribe.run(audio_file)
    return result

@svc.api(
    input=File(),
    output=JSON()
)
def transcribe(audio_file) -> dict:
    """Alias for transcribe_audio"""
    result = runner.transcribe.run(audio_file)
    return result

@svc.api(output=JSON())
def health() -> dict:
    """Health check"""
    return {"status": "healthy", "model": "whisper-small-twi"}
```

### Step 4: Create `bentofile.yaml`

```yaml
service: "twi_asr_service:svc"
labels:
  owner: "you"
  stage: "production"
python:
  requirements: |
    transformers==4.35.0
    librosa==0.10.0
    torch==2.0.0
    numpy==1.24.3
```

### Step 5: Run the Inference Server

```bash
bentoml serve twi_asr_service:TwiASRService --port 8001
```

You'll see:
```
 2024-11-07 10:30:00 INFO     [generated] Service 'twi_asr_inference' created successfully.
 2024-11-07 10:30:01 INFO     Service is running at http://127.0.0.1:8001
 2024-11-07 10:30:02 INFO     ✓ Model loaded
```

**Swagger UI available at:** `http://localhost:8001/docs`

---

## API Endpoints

- `POST /transcribe` - Upload audio file for transcription
- `POST /transcribe_audio` - Alias for transcribe endpoint  
- `GET /health` - Health check endpoint

**Swagger UI:** `http://localhost:8001/docs`

---

## Testing

```bash
# Test with curl
curl -X POST http://localhost:8001/transcribe \
  -F "file=@audio.wav"

# Response:
# {
#   "transcript": "hello world in twi"
# }

# Health check
curl http://localhost:8001/health
```

## Model

Uses `teckedd/whisper-small-serlabs-twi-asr` - a Whisper model fine-tuned for Twi (Akan) language speech recognition.