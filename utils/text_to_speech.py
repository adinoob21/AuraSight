
from murf import Murf
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

murf_api_key = os.getenv("MURF_API_KEY")


client = Murf(
    api_key=murf_api_key # Not required if you have set the MURF_API_KEY environment variable
)

def generate_voice(text:str):

    file_name = f"{uuid.uuid4().hex}.wav"
    file_path = f"speech_data/{file_name}"

    # Stream and save audio
    with open(file_path, "wb") as f:
        res = client.text_to_speech.stream(text=text, voice_id="en-US-terrell")
        for chunk in res:
            f.write(chunk)

    return file_path

