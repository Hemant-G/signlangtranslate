class TextBuffer:
    def __init__(self, debounce_frames=15):
        """
        Handles appending words natively, with debouncing for duplicates.
        """
        self.text = ""
        self.last_added = None
        self.cooldown = 0
        self.debounce_frames = debounce_frames
        
    def append(self, char):
        """
        Adds a character if it's not the same as the last added one OR after cooldown expires (for repeating letters like LL in HELLO).
        Actually, for sign language, standard repetition needs a hand-drop/transition.
        So we just block completely if the gesture hasn't transitioned, or maybe permit if the gesture transitions to Unknown and back.
        We'll treat "Unknown" predictions upstream as the cooldown tick factor or state breaker.
        """
        if char == "Unknown":
            # Just clear the block so the next letter can be appended
            self.last_added = None
            return False

        if char == self.last_added:
            return False
                
        # Register new character
        self.text += char
        self.last_added = char
        return True
        
    def space(self):
        if not self.text.endswith(" "):
            self.text += " "
            self.last_added = " "
            
    def backspace(self):
        if len(self.text) > 0:
            self.text = self.text[:-1]
            # Reset block so they can re-type the last letter immediately
            self.last_added = None 
            
    def clear(self):
        self.text = ""
        self.last_added = None
        
    def get_text(self):
        return self.text
