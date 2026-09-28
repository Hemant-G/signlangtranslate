import pyttsx3
import threading

class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
    def _speak_sync(self, text):
        if text.strip():
            self.engine.say(text)
            self.engine.runAndWait()
            
    def speak(self, text):
        """Non-blocking speak using a thread."""
        thread = threading.Thread(target=self._speak_sync, args=(text,))
        thread.start()
