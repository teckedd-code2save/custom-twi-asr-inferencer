import bentoml
import torch
import librosa
import tempfile
import os
import time
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Annotated
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conditional resources based on GPU availability
resources = {"gpu": "1"} if torch.cuda.is_available() else {"cpu": "2"}

@bentoml.service(
    resources=resources,
    traffic={"timeout": 60},
)
class TwiASRService:
    def __init__(self):
        start_time = time.time()
        print("Loading Twi ASR model...")
        
        self.processor = WhisperProcessor.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            device = "GPU"
        else:
            device = "CPU"
            
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded on {device} in {load_time:.2f}s")

    @bentoml.api
    def transcribe_audio(self, audio_file: Annotated[Path, bentoml.validators.ContentType("audio/*")]) -> dict:
        logger.info(f"Transcription request: {audio_file.name}")
        return self._transcribe_file(audio_file)
    
    @bentoml.api
    def transcribe(self, audio_file: Annotated[Path, bentoml.validators.ContentType("audio/*")]) -> dict:
        logger.info(f"Transcription request: {audio_file.name}")
        return self._transcribe_file(audio_file)
    
    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "model": "whisper-small-twi"}
    
    def _transcribe_file(self, audio_file: Path) -> dict:
        start_time = time.time()
        try:
            speech, sr = librosa.load(str(audio_file), sr=16000)
            
            with torch.no_grad():
                input_features = self.processor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                predicted_ids = self.model.generate(input_features)
            
            transcript = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            duration = time.time() - start_time
            logger.info(f"Transcription completed in {duration:.2f}s: {transcript[:50]}...")
            return {"transcript": transcript}
        
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file.name}: {str(e)}")
            return {"error": str(e)}