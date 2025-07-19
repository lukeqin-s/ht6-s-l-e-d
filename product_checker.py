import google.generativeai as genai
import pyttsx3
import speech_recognition as sr

# ========== Gemini API Setup ==========
genai.configure(api_key="AIzaSyBpAIh5RZTU6LRVmvdH-ekttbe4QZ32Enk")  
model = genai.GenerativeModel("models/gemini-1.5-flash")

# ========== Text-to-Speech ==========
tts = pyttsx3.init()
voices = tts.getProperty('voices')    
tts.setProperty('voice', voices[1].id)  # Use a specific voice
tts.setProperty('rate', 200)
tts.setProperty('volume', 0.9)

def speak(text):
    print(f"[TTS] {text}")
    tts = pyttsx3.init()
    tts.say(text)
    tts.runAndWait()

# ========== Voice Input ==========
def get_user_prompt_by_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ask your question or say 'I'm done' to exit...")
        speak("Ask your question about the item, or say 'I'm done' to finish.")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).strip().lower()
        print(f"[User]: {text}")
        return text
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that. Please say it again.")
        return None
    except sr.RequestError as e:
        print(f"[Speech Recognition Error]: {e}")
        speak("There was a speech service error.")
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

    speak(result)
    return result

# ========== Assistant Conversation Loop ==========
def run_farmers_market_assistant(product_name, price):
    speak(f"{product_name} detected, priced at {price}. You may now ask questions.")

    while True:
        user_question = get_user_prompt_by_voice()
        if user_question is None:
            continue

        if any(exit_phrase in user_question for exit_phrase in ["i'm done", "stop", "quit", "exit", "no more questions"]):
            speak("Alright! Ending session. Enjoy your time at the farmer's market.")
            break

        analyze_product(product_name, price, user_question)

# ========== Main Program ==========
if __name__ == "__main__":
    # Simulated product detection result (replace with your actual detection system output)
    detected_product = "Honeycrisp Apple"
    detected_price = "$2.50 per pound"

    print(f"[Detected Item]: {detected_product} | [Price]: {detected_price}")
    run_farmers_market_assistant(detected_product, detected_price)
