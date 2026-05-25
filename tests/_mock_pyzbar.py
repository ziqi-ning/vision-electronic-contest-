"""Mock hardware libs (pyzbar, apriltag, serial) not available on Windows dev machine."""
from unittest.mock import MagicMock
import sys

# --- pyzbar ---
class _MockRect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x; self.y = y; self.w = w; self.h = h

class _MockBarcode:
    def __init__(self, data=b'0', rect=None, btype='QRCODE'):
        self.data = data
        self.rect = rect or _MockRect()
        self.type = btype

def _decode(image, symbols=None):
    return []

class _Pyzbar:
    decode = staticmethod(_decode)

sys.modules['pyzbar'] = MagicMock()
sys.modules['pyzbar.pyzbar'] = _Pyzbar()

# --- apriltag ---
sys.modules['apriltag'] = MagicMock()

# --- serial ---
sys.modules['serial'] = MagicMock()
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()
sys.modules['serial.tools.list_ports_common'] = MagicMock()
sys.modules['serial.tools.list_ports_windows'] = MagicMock()
sys.modules['serial.win32'] = MagicMock()
