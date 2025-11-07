import bentoml
import torch
import librosa
import tempfile
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Annotated
from pathlib import Path

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class TwiASRService:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "teckedd/whisper-small-serlabs-twi-asr"
        )
        self.model.eval()
        print("✓ Model loaded")

    @bentoml.api
    def transcribe_audio(self, audio_file: Annotated[Path, bentoml.validators.ContentType("audio/*")]) -> dict:
        return self._transcribe_file(audio_file)
    
    @bentoml.api
    def transcribe(self, audio_file: Annotated[Path, bentoml.validators.ContentType("audio/*")]) -> dict:
        return self._transcribe_file(audio_file)
    
    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "model": "whisper-small-twi"}
    
    def _transcribe_file(self, audio_file: Path) -> dict:
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
            
            return {"transcript": transcript}
        
        except Exception as e:
            return {"error": str(e)}