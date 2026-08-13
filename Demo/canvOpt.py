from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from scipy.ndimage import center_of_mass, shift, zoom
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Reconocimiento de dígitos",
    page_icon="✏️",
    layout="centered",
)

st.title("Reconocimiento de dígitos ✏️🤖")
st.caption("Dibuja un número del 0 al 9 y el modelo intentará reconocerlo.")


# -----------------------------------------------------------------------------
# Modelo
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def find_model_path() -> Path | None:
    """Busca el modelo en ubicaciones habituales del proyecto."""
    candidates = (
        BASE_DIR / "mi_modelo.h5",
        BASE_DIR / "Demo" / "mi_modelo.h5",
        Path.cwd() / "mi_modelo.h5",
        Path.cwd() / "Demo" / "mi_modelo.h5",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


MODEL_PATH = find_model_path()


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: Path):
    return tf.keras.models.load_model(path, compile=False)


if MODEL_PATH is None:
    st.error("No se encontró el archivo mi_modelo.h5.")
    st.info(
        "Colócalo junto al archivo de la aplicación o dentro de una "
        "subcarpeta llamada Demo."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"No fue posible cargar el modelo: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Preprocesamiento tipo MNIST
# -----------------------------------------------------------------------------

def rgba_to_ink(image_rgba: np.ndarray) -> np.ndarray:
    """Convierte el canvas RGBA en intensidad de tinta: 0=fondo, 1=trazo."""
    if image_rgba.ndim != 3 or image_rgba.shape[-1] < 3:
        raise ValueError(f"Imagen de canvas no válida: {image_rgba.shape}")

    rgb = image_rgba[..., :3].astype(np.float32) / 255.0

    # Conversión perceptual a escala de grises.
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    return np.clip(1.0 - gray, 0.0, 1.0)


def has_drawing(ink: np.ndarray, threshold: float = 0.08) -> bool:
    """Evita interpretar ruido o un canvas vacío como un dígito."""
    return bool(np.count_nonzero(ink > threshold) >= 8)


def mnist_normalize(
    ink: np.ndarray,
    output_size: int = 28,
    digit_size: int = 20,
    threshold: float = 0.08,
) -> np.ndarray:
    """
    Recorta, escala y centra un trazo siguiendo la geometría típica de MNIST.

    El dígito ocupa como máximo 20x20 píxeles dentro de una imagen 28x28 y se
    desplaza para que su centro de masa quede en el centro de la imagen.
    """
    mask = ink > threshold
    if not np.any(mask):
        raise ValueError("El canvas no contiene un trazo reconocible.")

    rows, cols = np.where(mask)
    cropped = ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    height, width = cropped.shape
    scale = digit_size / max(height, width)
    resized = zoom(cropped, (scale, scale), order=1, prefilter=False)

    # zoom puede variar un píxel por redondeo.
    resized = resized[:digit_size, :digit_size]
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)

    canvas = np.zeros((output_size, output_size), dtype=np.float32)
    new_height, new_width = resized.shape
    top = (output_size - new_height) // 2
    left = (output_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized

    mass_center = center_of_mass(canvas)
    if np.all(np.isfinite(mass_center)):
        target = (output_size - 1) / 2.0
        displacement = (target - mass_center[0], target - mass_center[1])
        canvas = shift(
            canvas,
            shift=displacement,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def adapt_to_model(image_28: np.ndarray) -> np.ndarray:
    """Adapta una imagen 28x28 a la forma de entrada del modelo."""
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError("Este demo espera un modelo con una sola entrada.")

    if len(input_shape) == 4:
        # Canales al final: (None, 28, 28, 1)
        if input_shape[-1] in (1, None):
            batch = image_28[None, ..., None]
        # Canales al principio: (None, 1, 28, 28)
        elif input_shape[1] in (1, None):
            batch = image_28[None, None, ...]
        else:
            raise ValueError(f"Entrada de imagen no reconocida: {input_shape}")
    elif len(input_shape) == 3:
        batch = image_28[None, ...]
    elif len(input_shape) == 2:
        batch = image_28.reshape(1, -1)
    else:
        raise ValueError(f"Forma de entrada no reconocida: {input_shape}")

    return batch.astype(np.float32)


def as_probabilities(output: np.ndarray) -> np.ndarray:
    """Acepta tanto probabilidades como logits producidos por el modelo."""
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.size != 10:
        raise ValueError(
            f"El modelo produjo {values.size} valores; se esperaban 10 clases."
        )

    looks_like_probabilities = (
        np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.isclose(values.sum(), 1.0, atol=1e-3)
    )

    if looks_like_probabilities:
        probabilities = values
    else:
        stable = values - np.max(values)
        exponentials = np.exp(stable)
        probabilities = exponentials / exponentials.sum()

    return probabilities.astype(np.float32)


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------

if "clear_canvas_requested" not in st.session_state:
    st.session_state.clear_canvas_requested = False


def clear_canvas():
    """Solicita cargar un dibujo vacío sin desmontar el componente."""
    st.session_state.clear_canvas_requested = True


# initial_drawing actúa como un pulso: solamente se envía el dibujo vacío en
# la ejecución causada por el botón. La key permanece constante, de modo que
# el componente no se desmonta ni cambia de tamaño.
if st.session_state.clear_canvas_requested:
    initial_drawing = {"version": "4.4.0", "objects": []}
    st.session_state.clear_canvas_requested = False
else:
    initial_drawing = None

canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    initial_drawing=initial_drawing,
    display_toolbar=False,
    update_streamlit=True,
    key="digit_canvas",
)

st.button(
    "Limpiar",
    use_container_width=True,
    on_click=clear_canvas,
)

if canvas_result.image_data is not None:
    try:
        ink = rgba_to_ink(canvas_result.image_data)

        if has_drawing(ink):
            normalized = mnist_normalize(ink)
            model_input = adapt_to_model(normalized)
            raw_output = model.predict(model_input, verbose=0)

            if isinstance(raw_output, (list, tuple)):
                if len(raw_output) != 1:
                    raise ValueError("El demo espera un modelo con una sola salida.")
                raw_output = raw_output[0]

            probabilities = as_probabilities(raw_output)
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            top_three = np.argsort(probabilities)[::-1][:3]

            result_left, result_right = st.columns([1, 1.4])

            with result_left:
                st.subheader("Predicción")
                st.markdown(
                    f"<div style='font-size:6rem; line-height:1; "
                    f"font-weight:700; text-align:center; color:#e63946;'>"
                    f"{prediction}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:center'>Confianza: "
                    f"<strong>{confidence:.1%}</strong></p>",
                    unsafe_allow_html=True,
                )

            with result_right:
                st.subheader("Tres opciones principales")
                for digit in top_three:
                    probability = float(probabilities[digit])
                    st.write(f"**{digit}** — {probability:.1%}")
                    st.progress(probability)

            with st.expander("Ver la imagen que recibe el modelo"):
                st.image(
                    normalized,
                    caption="Imagen normalizada a 28 × 28",
                    clamp=True,
                    width=224,
                )

            if confidence < 0.60:
                st.info(
                    "La confianza es baja. Prueba dibujando el dígito más grande "
                    "y con un solo trazo continuo."
                )
        else:
            st.caption("La predicción aparecerá automáticamente al dibujar.")

    except Exception as exc:
        st.error(f"Error al procesar la predicción: {exc}")


with st.expander("Información técnica"):
    st.write(f"Modelo: `{MODEL_PATH.name}`")
    st.write(f"Forma de entrada: `{model.input_shape}`")
    st.write(f"Forma de salida: `{model.output_shape}`")
