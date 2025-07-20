import google.generativeai as genai2
# import pyttsx3
import speech_recognition as sr

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
        # save_wave_file(file_path, audio_data)

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

# ========== Gemini API Setup ==========
genai2.configure(api_key="AIzaSyAyN9cloankoPG3T9zCqT5ExVleXZJKocg")  
model = genai2.GenerativeModel("models/gemini-1.5-flash")

# ========== Text-to-Speech ==========
# tts = pyttsx3.init()
# voices = tts.getProperty('voices')    
# tts.setProperty('voice', voices[1].id)  # Use a specific voice
# tts.setProperty('rate', 200)
# tts.setProperty('volume', 0.9)

# def speak(text):
#     print(f"[TTS] {text}")
#     tts = pyttsx3.init()
#     tts.say(text)
#     tts.runAndWait()

# ========== Voice Input ==========
def get_user_prompt_by_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ask your question or say 'I'm done' to exit...")
        speak_text_bluetooth("Ask your question about the item, or say 'I'm done' to finish.")
        # speak("Ask your question about the item, or say 'I'm done' to finish.")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).strip().lower()
        print(f"[User]: {text}")
        return text
    except sr.UnknownValueError:
        speak_text_bluetooth("Sorry, I didn't catch that. Please say it again.")
        # speak("Sorry, I didn't catch that. Please say it again.")
        return None
    except sr.RequestError as e:
        print(f"[Speech Recognition Error]: {e}")
        speak_text_bluetooth("There was a speech service error.")
        # speak("There was a speech service error.")
        return None

# ========== Gemini Prompt Builder ==========
def build_prompt(product_name, price, user_question):
    return (
        f"The user is shopping at a farmer's market. "
        f"They're interested in: {product_name}, which costs {price}.\n\n"
        f"The user asked: \"{user_question}\"\n"
        f"Please respond in a clear, helpful tone, suitable for a shopping assistant. "
        f"Keep it simple and informative."
    )

# ========== Query Gemini + TTS Output ==========
def analyze_product(product_name, price, user_question):
    prompt = build_prompt(product_name, price, user_question)
    print(f"[Gemini Prompt]:\n{prompt}\n")

    response = model.generate_content(prompt)
    result = response.text.strip()

    speak_text_bluetooth(result, language_code="en-US", voice_name="Kore")
    # speak(result)
    return result

# ========== Assistant Conversation Loop ==========
def run_farmers_market_assistant(product_name, price):
    speak_text_bluetooth(f"{product_name} detected, priced at {price}. You may now ask questions.", language_code="en-US", voice_name="Kore")

    # speak(f"{product_name} detected, priced at {price}. You may now ask questions.")

    while True:
        user_question = get_user_prompt_by_voice()
        if user_question is None:
            continue

        if any(exit_phrase in user_question for exit_phrase in ["thank you", "i'm done", "stop", "quit", "exit", "no more questions"]):
            # speak("Alright! Ending session. Enjoy your time at the farmer's market.")
            speak_text_bluetooth("Alright! Ending session. Enjoy your time at the farmer's market.")
            break

        analyze_product(product_name, price, user_question)

# ========== Main Program ==========
# if __name__ == "__main__":
#     # Simulated product detection result (replace with your actual detection system output)
#     detected_product = "Honeycrisp Apple"
#     detected_price = "$2.50 per pound"

#     print(f"[Detected Item]: {detected_product} | [Price]: {detected_price}")
#     run_farmers_market_assistant(detected_product, detected_price)


# Call this function to start the voice assistant with a product name and price 
def start_voice_assistant(product_name, price):
    print(f"[Detected Item]: {product_name} | [Price]: {price}")
    run_farmers_market_assistant(product_name, price)

