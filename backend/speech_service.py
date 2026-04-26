import os
from openai import OpenAI

# Initialize the exact same AI client you are using for the rest of your app!
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

def transcribe_audio(audio_path):
    """
    Transcribes audio to text using the lightning-fast Groq Whisper API instead of a heavy local PyTorch model.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            # We use whisper-large-v3 model natively hosted on Groq
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="json"
            )
            return transcription.text.strip()
    except Exception as e:
        print(f"Transcription Error (Groq API): {e}")
        return "Error in audio transcription."