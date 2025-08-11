from pathlib import Path
import os

# ---------- CONFIG / CONSTANTS ----------
DPI = 300
A4_WIDTH = int(8.27 * DPI)   # 2481 px at 300 DPI
A4_HEIGHT = int(11.69 * DPI) # 3507 px at 300 DPI
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

# Model defaults (can override with env vars U2NET_MODEL & MODNET_MODEL)
U2NET_PATH = Path(os.getenv("U2NET_MODEL", r"C:\SnapSheet\models\u2net.onnx"))
MODNET_PATH = Path(os.getenv("MODNET_MODEL", r"C:\SnapSheet\models\modnet_photographic_portrait_matting.onnx"))
