import cv2
import numpy as np
import onnxruntime as ort
from io import BytesIO
from config import U2NET_PATH, MODNET_PATH
from image_utils import load_cv_image, cv2_to_rgb_for_model

# ONNX sessions (lazy init)
u2net_session = None
modnet_session = None

def ensure_models_loaded():
    """Load ONNX sessions the first time this is called."""
    global u2net_session, modnet_session
    if u2net_session is None:
        if not U2NET_PATH.is_file():
            raise FileNotFoundError(f"U2Net model not found: {U2NET_PATH}")
        u2net_session = ort.InferenceSession(str(U2NET_PATH))

    if modnet_session is None:
        if not MODNET_PATH.is_file():
            raise FileNotFoundError(f"MODNet model not found: {MODNET_PATH}")
        modnet_session = ort.InferenceSession(str(MODNET_PATH))

def remove_background(image_input, input_size=(320, 320), return_mask=False, top_margin_percent=0.10):
    """
    Remove background using U2Net + MODNet combination.
    Returns PNG bytes (RGBA) or mask bytes when return_mask=True.
    """
    ensure_models_loaded()

    img = load_cv_image(image_input)
    if img is None:
        raise ValueError("Failed to load image.")

    original_bgr = img.copy()
    if original_bgr.ndim == 2:
        original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_GRAY2BGR)
    elif original_bgr.shape[2] == 4:
        original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_BGRA2BGR)

    image_rgb = cv2_to_rgb_for_model(img)
    original_h, original_w = image_rgb.shape[:2]

    # --- U2Net pass ---
    resized = cv2.resize(image_rgb, input_size)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    input_tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)

    u2net_out = u2net_session.run([u2net_session.get_outputs()[0].name],
                                  {u2net_session.get_inputs()[0].name: input_tensor})[0]
    u2_mask = u2net_out.squeeze()
    u2_mask = cv2.resize(u2_mask, (original_w, original_h))
    u2_mask = (u2_mask * 255).astype(np.uint8)

    # --- MODNet pass ---
    resized_mod = cv2.resize(image_rgb, (512, 512))
    resized_mod = resized_mod.astype(np.float32) / 255.0
    input_mod = np.expand_dims(np.transpose(resized_mod, (2, 0, 1)), axis=0)

    modnet_out = modnet_session.run([modnet_session.get_outputs()[0].name],
                                    {modnet_session.get_inputs()[0].name: input_mod})[0]
    mod_mask = modnet_out.squeeze()
    mod_mask = cv2.resize(mod_mask, (original_w, original_h))
    mod_mask = (mod_mask * 255).astype(np.uint8)

    # Combine & refine
    _, u2_t = cv2.threshold(u2_mask, 128, 255, cv2.THRESH_BINARY)
    _, mod_t = cv2.threshold(mod_mask, 128, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(u2_t, mod_t)
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.GaussianBlur(refined, (5, 5), 0)

    if return_mask:
        success, buf = cv2.imencode('.png', final_mask)
        if not success:
            raise ValueError("Failed to encode mask.")
        return buf.tobytes()

    result_bgra = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
    result_bgra[:, :, 3] = final_mask

    _, thresh = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No foreground detected in the image.")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    extra = int(original_h * top_margin_percent)
    new_h = h + extra
    new_w = w

    final_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    final_image[extra:, :, :] = result_bgra[y:y+h, x:x+w]

    success, buf = cv2.imencode('.png', final_image)
    if not success:
        raise ValueError("Failed to encode output image.")
    return buf.tobytes()
