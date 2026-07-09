"""Streamlit demo for the IAM handwritten-word OCR model.

Thin UI layer only — all model/inference logic below is copied verbatim from
paint.py (which cannot be imported here because it initializes a pygame window
at import time).
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras import backend as K

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

REPO_ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = "best_model.keras"

# ---------------------------------------------------------------------------
# Core logic from paint.py — unchanged
# ---------------------------------------------------------------------------

chars = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def process_image(img):
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)
    img = img / 255.0
    return img

def predict_word(loaded_model, img):
    """Prediction + CTC decode, as in paint.py's predict_func."""
    img = process_image(img)
    img = np.expand_dims(img, axis=0)

    prediction = loaded_model.predict(img, verbose=0)
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded = K.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
    output_text = ''.join([chars[int(i)] for i in decoded[0] if int(i) != -1])
    return output_text

# ---------------------------------------------------------------------------
# Model loading (repo-root-relative, with optional Hugging Face Hub fallback)
# ---------------------------------------------------------------------------

def _resolve_model_path():
    local = REPO_ROOT / MODEL_FILENAME
    # A real model is ~100MB; a Git LFS pointer checkout is a few hundred bytes.
    if local.exists() and local.stat().st_size > 1_000_000:
        return str(local)

    hf_repo = ""
    try:
        hf_repo = st.secrets.get("HF_MODEL_REPO", "")
    except Exception:
        pass
    if hf_repo:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=hf_repo, filename=MODEL_FILENAME)

    st.error(
        f"`{MODEL_FILENAME}` was not found (or is an unresolved Git LFS pointer). "
        "Either commit the real weights file to the repo, or upload it to a "
        "Hugging Face Hub repo and set `HF_MODEL_REPO = \"<user>/<repo>\"` in "
        "the app's Streamlit secrets."
    )
    st.stop()

@st.cache_resource(show_spinner="Loading OCR model…")
def load_ocr_model():
    return keras.models.load_model(
        _resolve_model_path(), custom_objects={'ctc_loss': ctc_loss}
    )

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Handwritten Word OCR", page_icon="✍️")
st.title("✍️ Handwritten Word Recognition")
st.write(
    "A CRNN + CTC model trained on the IAM words dataset reads a single "
    "handwritten English word. Draw a word below or upload a picture of one, "
    "then hit **Predict**."
)

loaded_model = load_ocr_model()

draw_tab, upload_tab = st.tabs(["Draw a word", "Upload an image"])

gray = None

with draw_tab:
    if HAS_CANVAS:
        brush_size = st.slider("Brush size", 2, 32, 8)
        canvas_result = st_canvas(
            stroke_width=brush_size * 2,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=600,
            height=200,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result is not None and canvas_result.image_data is not None:
            rgba = canvas_result.image_data.astype(np.uint8)
            alpha = rgba[:, :, 3:4] / 255.0
            rgb = (rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
            drawn = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if drawn.min() < 200:  # something was actually drawn
                # paint.py downscales the 600x200 canvas to 300x100 before predicting
                gray = cv2.resize(drawn, (300, 100), interpolation=cv2.INTER_AREA)
    else:
        st.info(
            "The drawing canvas component (`streamlit-drawable-canvas`) is not "
            "available — use the **Upload an image** tab instead."
        )

with upload_tab:
    uploaded = st.file_uploader(
        "Image of a single handwritten word (dark ink on a light background)",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded is not None:
        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            st.error("Could not read that file as an image.")
        else:
            st.image(gray, caption="Uploaded image", clamp=True)

if st.button("Predict", type="primary", disabled=gray is None):
    with st.spinner("Recognizing…"):
        word = predict_word(loaded_model, gray)
    if word:
        st.success(f"**Predicted word:** `{word}`")
    else:
        st.warning("The model returned an empty prediction — try writing more clearly or larger.")
elif gray is None:
    st.caption("Draw or upload a word to enable prediction.")
