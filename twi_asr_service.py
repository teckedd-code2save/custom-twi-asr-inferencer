import bentoml
import torch
import librosa
import time
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import transformers
transformers.logging.set_verbosity_error()
from typing import Annotated
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine compute resources
HAS_CUDA = torch.cuda.is_available()
RESOURCES = {"gpu": 1} if HAS_CUDA else {"cpu": "2"}

@bentoml.service(
    name="twi_asr_inference",
    resources=RESOURCES,
    traffic={
        "timeout": 60,
        "max_concurrency": 10,
    },
)
class TwiASRService:
    def __init__(self):
        start_time = time.time()
        logger.info("Initializing Twi ASR Service...")
        
        model_id = "teckedd/whisper-small-serlabs-twi-asr"
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        # Optimization: Move to GPU and use half-precision if available
        if HAS_CUDA:
            self.model = self.model.to("cuda").half()
            self.device = "cuda"
            logger.info("Using GPU with FP16 precision")
        else:
            self.device = "cpu"
            logger.info("Using CPU with FP32 precision")
            
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")

    @bentoml.api
    def transcribe(self, audio_file: Annotated[Path, bentoml.validators.ContentType("audio/*")]) -> dict:
        """
        Transcribes an uploaded audio file into Twi text.
        """
        logger.info(f"Processing transcription request: {audio_file.name}")
        return self._run_inference(audio_file)

    @bentoml.api
    def health(self) -> dict:
        """
        Detailed health check including device status.
        """
        return {
            "status": "healthy",
            "model": "whisper-small-twi",
            "device": self.device,
            "torch_version": torch.__version__
        }
    
    def _run_inference(self, audio_path: Path) -> dict:
        start_time = time.time()
        try:
            # Load and resample audio
            speech, sr = librosa.load(str(audio_path), sr=16000)
            audio_duration = len(speech) / sr
            
            with torch.no_grad():
                # Pre-process
                input_features = self.processor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                # Move features to same device as model
                if HAS_CUDA:
                    input_features = input_features.to("cuda").half()
                
                # Generate IDs
                infer_time_start = time.time()
                predicted_ids = self.model.generate(input_features)
                inference_time_sec = time.time() - infer_time_start
            
            # Decode to text
            transcript = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            processing_time = time.time() - start_time
            logger.info(f"✓ Success: {audio_path.name} ({audio_duration:.1f}s audio) in {processing_time:.2f}s (inference: {inference_time_sec:.2f}s)")
            
            return {
                "transcript": transcript.strip(),
                "metadata": {
                    "audio_duration_sec": round(audio_duration, 2),
                    "processing_time_sec": round(processing_time, 2),
                    "inference_time_sec": round(inference_time_sec, 2),
                    "rtf": round(processing_time / audio_duration, 3) if audio_duration > 0 else 0,
                    "device": self.device
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {str(e)}", exc_info=True)
            return {
                "error": "Transcription failed",
                "message": str(e)
            }