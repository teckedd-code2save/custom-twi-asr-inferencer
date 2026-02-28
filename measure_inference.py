import requests
import time
import numpy as np
import scipy.io.wavfile as wavfile
import io

# Generate a 3-second 16kHz dummy audio to benchmark
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio_data = np.sin(2 * np.pi * 300 * t) * 0.5
audio_data = (audio_data * 32767).astype(np.int16)

virt_file = io.BytesIO()
wavfile.write(virt_file, sample_rate, audio_data)
virt_file.seek(0)

print(f"Testing local inference server with {duration}s generated audio frame...")

start_req = time.time()
try:
    response = requests.post("http://localhost:8001/transcribe", files={"audio_file": ("test.wav", virt_file, "audio/wav")})
    total_time = time.time() - start_req
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- BENCHMARK RESULTS ---")
        print(f"Total Client Round-Trip Time: {total_time:.2f} seconds")
        print(f"Server Processing Time:       {data['metadata']['processing_time_sec']} seconds")
        print(f"Pure Model Inference Time:    {data['metadata']['inference_time_sec']} seconds")
        print(f"Real-Time Factor (RTF):       {data['metadata']['rtf']}")
        print(f"Compute Device:               {data['metadata']['device']}")
    else:
        print(f"Failed! Code: {response.status_code}, Text: {response.text}")
except Exception as e:
    print("Error reaching server:", e)
