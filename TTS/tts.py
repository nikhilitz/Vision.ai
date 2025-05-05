# tts.py

# Option 1: Offline TTS using pyttsx3
import pyttsx3

def speak_offline(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)       # Speed of speech
    engine.setProperty('volume', 1.0)     # Volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()


# Option 2: Online TTS using Google Text-to-Speech (gTTS)
from gtts import gTTS
import os
import tempfile

def speak_online(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            os.system(f"start {fp.name}" if os.name == 'nt' else f"afplay {fp.name}")  # 'afplay' for macOS, 'start' for Windows
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
