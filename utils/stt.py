# Install the assemblyai package by executing the command "pip install assemblyai"

# import assemblyai as aai

# aai.settings.api_key = ""

# # audio_file = "./local_file.mp3"
# audio_file = "https://assembly.ai/wildfires.mp3"

# config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

# transcript = aai.Transcriber(config=config).transcribe(audio_file)

# if transcript.status == "error":
#   raise RuntimeError(f"Transcription failed: {transcript.error}")

# print(transcript.text)

import assemblyai as aai
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("STT_API_KEY")

def transcribe_audio_to_text(audio_path: str) -> str:
    """
    Transcribes a local audio file to text using AssemblyAI.

    Args:
        audio_path (str): Path to the .wav audio file.
        api_key (str): Your AssemblyAI API key.

    Returns:
        str: Transcribed text.
    """
    # Set the API key
    aai.settings.api_key = api_key

    # Transcription configuration (optional)
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

    # Transcribe the audio
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text
