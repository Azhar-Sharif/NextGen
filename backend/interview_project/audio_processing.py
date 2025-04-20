import asyncio
import sounddevice as sd
import numpy as np
import wave
import tempfile
import time
import aiofiles
import sys
from pathlib import Path
from typing import Optional, Tuple
from openai import AsyncOpenAI

async def record_audio_async(fs: int = 16000, timeout: int = 30) -> Optional[np.ndarray]:
    """
    Records audio for a fixed amount of time or until Enter is pressed.
    
    Args:
        fs (int): Sampling frequency, default 16000 Hz
        timeout (int): Maximum recording time in seconds, default 30 seconds
        
    Returns:
        Optional[np.ndarray]: The recorded audio data as a numpy array,
                             or None if no audio was recorded
    """
    print("Recording... Press Enter to stop or wait for timeout.")

    def input_thread(stop_event):
        """Thread function that waits for Enter key and sets stop event."""
        input()  # Wait for Enter key press
        stop_event.set()

    def record_stream(stop_event) -> Optional[np.ndarray]:
        """Records audio from the stream until stop event is set."""
        audio_data = []
        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
                start_time = time.time()
                while not stop_event.is_set():
                    frame, overflowed = stream.read(1024)
                    if overflowed:
                        print('Audio buffer overflow detected', file=sys.stderr)
                    audio_data.append(frame)
                    if time.time() - start_time > timeout:
                        print("Timeout reached. Stopping recording.")
                        break
                        
            # Process the recorded data
            if not audio_data:
                return None
                
            return np.concatenate(audio_data, axis=0)
        except Exception as e:
            print(f"Error during audio recording: {e}", file=sys.stderr)
            return None

    try:
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        
        # Start the input listener
        loop.run_in_executor(None, input_thread, stop_event)
        
        # Start recording
        audio = await loop.run_in_executor(None, record_stream, stop_event)
        
        # Log audio information
        if audio is not None:
            print(f"Debug: Audio shape: {audio.shape}", file=sys.stderr)
        else:
            print("Debug: No audio data captured", file=sys.stderr)
        
        return audio
    except Exception as e:
        print(f"Error in record_audio_async: {e}", file=sys.stderr)
        return None


async def save_audio_to_wav_async(audio_data: np.ndarray, fs: int = 16000) -> Optional[str]:
    """
    Saves audio data to a temporary WAV file.
    
    Args:
        audio_data (np.ndarray): The audio data to save
        fs (int): Sampling frequency, default 16000 Hz
        
    Returns:
        Optional[str]: Path to the temporary WAV file, or None if saving failed
    """
    if audio_data is None:
        print("Error: No audio data to save", file=sys.stderr)
        return "no audio recorded"
        
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
            tmpfile_name = tmpfile.name
            
            def write_wav():
                """Writes audio data to WAV file."""
                try:
                    with wave.open(tmpfile_name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(fs)
                        wf.writeframes(audio_data.tobytes())
                except Exception as e:
                    print(f"Error writing WAV file: {e}", file=sys.stderr)
                    raise
                    
            # Write audio data to file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, write_wav)
            return tmpfile_name
    except Exception as e:
        print(f"Error saving audio to WAV: {e}", file=sys.stderr)
        return None


async def transcribe_audio_with_whisper_async(audio_file: str, api_key: str) -> Optional[str]:
    """
    Transcribes audio using OpenAI's Whisper model.
    
    Args:
        audio_file (str): Path to the audio file to transcribe
        api_key (str): OpenAI API key
        
    Returns:
        Optional[str]: The transcribed text, or None if transcription failed
    """
    if not audio_file or not api_key:
        print("Error: Missing audio file or API key", file=sys.stderr)
        return "missing audio file"
        
    client = AsyncOpenAI(api_key=api_key)
    try:
        # Convert string path to Path object
        file_path = Path(audio_file)
        if not file_path.exists():
            print(f"Error: Audio file not found at {audio_file}", file=sys.stderr)
            return "missing audio file"
            
        # Send to OpenAI for transcription
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=file_path,
            language="en"
        )
        
        # Return transcribed text
        return response.text
    except Exception as e:
        print(f"Error transcribing audio with Whisper: {e}", file=sys.stderr)
        return "missing audio file"