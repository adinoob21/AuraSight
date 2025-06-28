import sounddevice as sd
from scipy.io.wavfile import write
import os
from datetime import datetime

def record_audio(output_dir: str = "recordings", duration: int = 5, sample_rate: int = 16000) -> str:
    """
    Records audio from mic and saves it to a .wav file in the given directory.

    Args:
        output_dir (str): Directory to save the audio file.
        duration (int): Duration of recording in seconds.
        sample_rate (int): Sample rate in Hz.

    Returns:
        str: Full path to the saved .wav file.
    """
    print(f"[Mic] Recording for {duration} seconds...")

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"voice_{timestamp}.wav")

    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        write(file_path, sample_rate, recording)
    except Exception as e:
        raise RuntimeError(f"[Mic] Recording failed: {e}")

    print(f"[Mic] Audio saved to {file_path}")
    return file_path
