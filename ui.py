import gradio as gr
import requests
import json
import io
import time
import os

PORT = int(os.environ.get("PORT", "8001"))
BENTOML_URL = f"http://127.0.0.1:{PORT}/transcribe"

def run_transcription(audio_path):
    if audio_path is None:
        return "Please upload or record audio.", ""

    try:
        # Load file directly as bytes using built-in open
        with open(audio_path, "rb") as f:
            files = {"audio_file": ("audio.wav", f, "audio/wav")}
            
            # Send to BentoML logic
            response = requests.post(BENTOML_URL, files=files, timeout=120)

        # Handle HTTP errors nicely
        if response.status_code != 200:
            error_msg = f"Error {response.status_code}: {response.text}"
            return "Transcription failed...", error_msg
        
        # Parse Response
        data = response.json()
        transcript = data.get("transcript", "No transcript found in response.")
        
        # Meta info
        meta = data.get("metadata", {})
        meta_str = json.dumps(meta, indent=2) if meta else "No metadata returned."
        
        return transcript, meta_str
        
    except requests.exceptions.ConnectionError:
        return "Connection Error", f"Failed to connect to BentoML server at {BENTOML_URL}. Is it running?"
    except Exception as e:
        return "Error occurred", str(e)


custom_css = """
body {
    background-color: #0b0f19;
    color: #e0e6ed;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 900px !important;
    margin: 40px auto;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
}
h1 {
    color: #3b82f6;
    text-align: center;
    font-size: 2.2rem;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 25px;
}
.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}
.footer {
    text-align: center;
    margin-top: 30px;
    font-size: 0.9rem;
    color: #6b7280;
}
"""

with gr.Blocks(css=custom_css, title="Twi ASR - Serlabs") as demo:
    gr.HTML("<h1>🎙️ Twi (Akan) Speech Recognition</h1>")
    gr.HTML("<p class='subtitle'>Powered by fine-tuned Whisper & BentoML Server</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio (Twi)")
            submit_btn = gr.Button("Transcribe Audio ✨", elem_classes=["primary-btn"], size="lg")
            
        with gr.Column(scale=1):
            transcript_out = gr.Textbox(label="Transcription", lines=5, placeholder="The transcribed Twi text will appear here...", interactive=False)
            with gr.Accordion("Advanced Metadata", open=False):
                meta_out = gr.Code(label="Response Metadata", language="json", interactive=False)
    
    submit_btn.click(
        fn=run_transcription,
        inputs=[audio_in],
        outputs=[transcript_out, meta_out],
        api_name="transcribe"
    )
    
    gr.HTML("<div class='footer'>Developed by Teckedd / Serlabs • Powered by <a href='https://bentoml.com' target='_blank' style='color:#3b82f6'>BentoML</a> & <a href='https://gradio.app' target='_blank' style='color:#3b82f6'>Gradio</a></div>")

# For running as a standalone script
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
