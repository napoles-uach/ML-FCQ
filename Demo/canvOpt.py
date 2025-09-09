import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from streamlit_drawable_canvas import st_canvas
from pathlib import Path

st.set_page_config(page_title="Digit Recognition App", page_icon="✏️")
st.markdown("# Digit :blue[Recognition] :green[App] ✏️🤖🔢")

# --------- CARGA DEL MODELO (.h5) ---------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

MODEL_PATH = Path("Demo/mi_modelo.h5")
if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en {MODEL_PATH.resolve()}")
    st.stop()

model = load_model(str(MODEL_PATH))

# --------- PREPROCESADO DE IMAGEN ---------
def process_image_rgba(x_rgba: np.ndarray, size=28) -> np.ndarray:
    """
    Convierte la imagen RGBA del canvas en la forma esperada por el modelo.
    - Pasa a escala de grises
    - Redimensiona a (28,28)
    - Normaliza a [0,1]
    - Invierte (1-x) para fondo negro y dígito blanco
    - Ajusta shape según input_shape
    """
    rgb = x_rgba[..., :3]          # tomar solo RGB
    gray = rgb.mean(axis=2)        # gris
    gray = gray.astype(np.float32) / 255.0

    # Resize con zoom
    h, w = gray.shape
    zoom_y, zoom_x = size / h, size / w
    gray_resized = zoom(gray, (zoom_y, zoom_x), order=1)

    # Invertir (fondo negro, dígito blanco típico de MNIST)
    gray_resized = 1.0 - gray_resized

    # Adaptar al input del modelo
    if len(model.input_shape) == 4:     # (None, 28, 28, 1)
        x = gray_resized[None, ..., None]
    else:                               # (None, 784)
        x = gray_resized.reshape(1, -1)

    return x

# --------- UI ---------
st.write("✍️ Draw a digit below:")

canvas_result = st_canvas(
    stroke_width=10,
    height=28 * 5,
    width=28 * 5,
    background_color="white",
    stroke_color="black",
    drawing_mode="freedraw",
    key="canvas",
)

st.header("Prediction:")

if canvas_result is not None and canvas_result.image_data is not None:
    img = canvas_result.image_data
    if np.sum(img) > 0:  # hay algo dibujado
        try:
            x = process_image_rgba(img, size=28)
            probs = model.predict(x)
            pred = int(np.argmax(probs, axis=1)[0])
            st.markdown(f"This number seems to be:\n\n# :red[{pred}]")
        except Exception as e:
            st.error(f"⚠️ Error procesando o prediciendo: {e}")
    else:
        st.write("No number drawn, please draw a digit.")
else:
    st.write("No number drawn, please draw a digit.")
