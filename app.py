import os
import google.generativeai as genai
import dotenv
import speech_recognition as sr
import streamlit as st
from gtts import gTTS

dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generation_config = {
    "temperature": 0.2, #HIGHER TEMP. MORE RANDOM OUTPUT(control randomness)
    "top_p": 0.8,   #HIGHER P VALUE MORE DIVERSIFY OUTPUT(select tokens based on cumulative probabilities
    "top_k": 45,    #HIGHER K VALUE MORE DIVERSIFY OUTPUT
    "max_output_tokens": 1024
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Language code table
language_codes = {
    "English": "en",
    "French": "fr",
    "Hindi": "hi",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Arabic": "ar",
    "Bengali": "bn",
    "Portuguese": "pt-BR",
    "Dutch": "nl",
    "Filipino": "fil",
    "Gujarati": "gu",
    "Italian": "it",
    "Kannada": "kn",
    "Korean": "ko",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Russian": "ru",
    "Chinese": "zh-CN",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Urdu": "ur"
}


def translate_text(text, source_language, target_language):
    prompt = f"Translate the following text from {source_language} to {target_language}:\"{text}\""
    response = model.generate_content(prompt)
    try:
        return response.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return None

def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak your text...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write("You said: " + text)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None


def generate_speech(text, language_code):
    tts = gTTS(text, lang=language_code)
    tts.save("translated_audio.mp3")
    with open("translated_audio.mp3", "rb") as audio_file:
        st.audio(audio_file, format="audio/mp3")


def main():
    st.title("LANGUAGE TRANSLATOR")
    input_option = st.radio("Input Method", ("Text Input", "Microphone Input"))

    if input_option == "Text Input":
        source_text = st.text_input("Enter text to translate:")

    else:
        source_text = get_audio_input()

    source_lang = st.selectbox("Source Language", ["Hindi","English","Gujarati","Marathi","Japanese","Korean","Urdu","Arabic","Bengali","Chinese","Dutch","French","Filipino","German","Italian","Kannada","Malayalam","Portuguese","Russian","Spanish","Swedish","Tamil","Telugu","Thai"])
    target_lang = st.selectbox("Target Language", ["Hindi","English","Gujarati","Marathi","Japanese","Korean","Urdu","Arabic","Bengali","Chinese","Dutch","French","Filipino","German","Italian","Kannada","Malayalam","Portuguese","Russian","Spanish","Swedish","Tamil","Telugu","Thai"])

    if st.button("Translate"):
        if source_text is not None:
            with st.spinner("Translating..."):
                translated_text = translate_text(source_text, source_lang, target_lang)
            st.write(f"Translated text:\n{translated_text}")

            target_lang_code = language_codes[target_lang]
            tts = gTTS(translated_text, lang=target_lang_code)
            tts.save("translated_audio.mp3")
            with open("translated_audio.mp3", "rb") as audio_file:
                st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
