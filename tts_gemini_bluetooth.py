from google import genai
from google.genai import types
import wave
import tempfile
import subprocess
import os

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyBqjlDwjCxz0v9udakz0iurSjjvZOYpDtM")

# Save wave file
def save_wave_file(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)

# Main function with multilingual support
def speak_text_bluetooth(text, language_code="en-US", voice_name="Kore"):
    # Generate TTS audio
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=language_code,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        )
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data

    # Save as temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_path = tmp.name
        save_wave_file(file_path, audio_data)

    print(f"[INFO] Audio saved to: {file_path}")

    # Play audio on Bluetooth speaker
    try:
        if os.name == 'posix':  # Linux/macOS
            subprocess.run(["aplay", file_path], check=True)
        elif os.name == 'nt':  # Windows
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        else:
            print("[WARNING] Unsupported OS for playback.")
    except Exception as e:
        print(f"[ERROR] Audio playback failed: {e}")

    os.remove(file_path)
