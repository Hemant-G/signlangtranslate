import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_buffer import TextBuffer

def test_text_buffer():
    buf = TextBuffer()
    
    # Test append
    assert buf.append("A") == True
    assert buf.get_text() == "A"
    
    # Test duplicate blocking
    assert buf.append("A") == False
    assert buf.get_text() == "A"
    
    # Test reset state block via Unknown
    buf.append("Unknown")
    assert buf.append("A") == True
    assert buf.get_text() == "AA"
    
    # Test Space
    buf.space()
    assert buf.get_text() == "AA "
    
    # Test Backspace
    buf.backspace()
    assert buf.get_text() == "AA"
    
    # Test Clear
    buf.clear()
    assert buf.get_text() == ""
