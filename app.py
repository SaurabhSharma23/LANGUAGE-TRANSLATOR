import os
import google.generativeai as genai
import dotenv
import streamlit as st
from gtts import gTTS
import whisper
import tempfile

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
whisper_model = whisper.load_model("base")
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
        audio_bytes = st.audio_input("Speak your text...")
        if audio_bytes:

            import numpy as np
            audio_data = np.frombuffer(audio_bytes.read(), dtype=np.float32)
            print(f"Audio data shape: {audio_data.shape}") 
            result = whisper_model.transcribe(audio_data)
            source_text = result["text"]

            st.write("You said: " + source_text)
        else:
            st.error("No audio input received. Please try again.")
            return
        

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
