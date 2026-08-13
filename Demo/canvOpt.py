from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from scipy.ndimage import center_of_mass, shift, zoom
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Reconocimiento de dígitos",
    page_icon="✏️",
    layout="wide",
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
# Espacio latente
# -----------------------------------------------------------------------------

def adapt_mnist_batch(images: np.ndarray, input_shape) -> np.ndarray:
    """Adapta un lote (N, 28, 28) a la entrada declarada por el modelo."""
    images = images.astype(np.float32)

    if isinstance(input_shape, list):
        raise ValueError("El espacio latente requiere un modelo de una entrada.")

    if len(input_shape) == 4:
        if input_shape[-1] in (1, None):
            batch = images[..., None]
        elif input_shape[1] in (1, None):
            batch = images[:, None, ...]
        else:
            raise ValueError(f"Entrada de imagen no reconocida: {input_shape}")
    elif len(input_shape) == 3:
        batch = images
    elif len(input_shape) == 2:
        batch = images.reshape(len(images), -1)
    else:
        raise ValueError(f"Forma de entrada no reconocida: {input_shape}")

    return batch.astype(np.float32)


@st.cache_resource(show_spinner="Construyendo el mapa del espacio latente...")
def build_latent_reference(
    _model,
    model_path: str,
    model_mtime_ns: int,
    samples_per_digit: int = 200,
):
    """Calcula y conserva una proyección PCA de la capa oculta del modelo."""
    # path y mtime forman parte de la llave del caché y fuerzan la
    # reconstrucción si cambia el archivo del modelo.
    _ = (model_path, model_mtime_ns)

    if len(_model.layers) < 2:
        raise ValueError("El modelo no contiene una capa oculta utilizable.")

    latent_layer = _model.layers[-2]
    latent_model = tf.keras.Model(
        inputs=_model.inputs,
        outputs=latent_layer.output,
        name="extractor_latente",
    )

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    selected = np.concatenate(
        [np.flatnonzero(y_test == digit)[:samples_per_digit] for digit in range(10)]
    )
    images = x_test[selected].astype(np.float32) / 255.0
    labels = y_test[selected].astype(np.int8)
    reference_input = adapt_mnist_batch(images, _model.input_shape)

    embeddings = latent_model.predict(
        reference_input,
        batch_size=256,
        verbose=0,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(len(selected), -1)

    latent_mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - latent_mean
    _, singular_values, components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    components = components[:2].astype(np.float32)
    coordinates = centered @ components.T

    total_variance = float(np.sum(singular_values**2))
    explained_variance = (
        singular_values[:2] ** 2 / total_variance
        if total_variance > 0
        else np.zeros(2)
    )

    return {
        "extractor": latent_model,
        "layer_name": latent_layer.name,
        "latent_dimensions": int(embeddings.shape[1]),
        "mean": latent_mean.astype(np.float32),
        "components": components,
        "coordinates": coordinates.astype(np.float32),
        "labels": labels,
        "explained_variance": np.asarray(explained_variance, dtype=np.float32),
    }


def render_latent_map(
    reference: dict,
    point: np.ndarray | None = None,
    prediction: int | None = None,
):
    """Dibuja MNIST y, si existe, resalta la posición del usuario."""
    coordinates = reference["coordinates"]
    labels = reference["labels"]

    reference_frame = pd.DataFrame(
        {
            "Componente 1": coordinates[:, 0],
            "Componente 2": coordinates[:, 1],
            "Dígito": labels.astype(str),
        }
    )
    background = (
        alt.Chart(reference_frame)
        .mark_circle(size=34, opacity=0.34)
        .encode(
            x=alt.X("Componente 1:Q", title="Primera componente principal"),
            y=alt.Y("Componente 2:Q", title="Segunda componente principal"),
            color=alt.Color(
                "Dígito:N",
                title="Dígito real",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=["Dígito:N", "Componente 1:Q", "Componente 2:Q"],
        )
    )

    chart = background

    if point is not None and prediction is not None:
        point_frame = pd.DataFrame(
            {
                "Componente 1": [float(point[0])],
                "Componente 2": [float(point[1])],
                "Predicción": [str(prediction)],
                "Etiqueta": [f"Tu dibujo: {prediction}"],
            }
        )

        user_point = (
            alt.Chart(point_frame)
            .mark_point(
                shape="diamond",
                filled=True,
                size=330,
                color="#111827",
                stroke="white",
                strokeWidth=2,
            )
            .encode(
                x="Componente 1:Q",
                y="Componente 2:Q",
                tooltip=["Etiqueta:N", "Componente 1:Q", "Componente 2:Q"],
            )
        )

        user_label = (
            alt.Chart(point_frame)
            .mark_text(dy=-18, fontSize=14, fontWeight="bold", color="#111827")
            .encode(
                x="Componente 1:Q",
                y="Componente 2:Q",
                text="Etiqueta:N",
            )
        )
        chart = chart + user_point + user_label

    chart = chart.properties(height=460).interactive()
    st.altair_chart(chart, use_container_width=True)


def show_representation_space(
    model_input: np.ndarray | None = None,
    prediction: int | None = None,
):
    """Muestra el mapa base y añade el dibujo cuando está disponible."""
    st.subheader("Espacio de representación")
    if model_input is None:
        st.caption(
            "Los puntos representan imágenes de MNIST. Dibuja un número para "
            "agregar su posición al mapa."
        )
    else:
        st.caption(
            "Los puntos representan imágenes de MNIST. El diamante negro indica "
            "dónde ubica la red el dígito que acabas de dibujar."
        )

    try:
        latent_reference = build_latent_reference(
            model,
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime_ns,
        )
        latent_point = None

        if model_input is not None:
            latent_vector = latent_reference["extractor"].predict(
                model_input,
                verbose=0,
            )
            latent_vector = np.asarray(
                latent_vector,
                dtype=np.float32,
            ).reshape(1, -1)
            latent_point = (
                latent_vector - latent_reference["mean"]
            ) @ latent_reference["components"].T

        render_latent_map(
            latent_reference,
            None if latent_point is None else latent_point[0],
            prediction,
        )

        explained = latent_reference["explained_variance"]
        st.caption(
            f"Capa `{latent_reference['layer_name']}`: "
            f"{latent_reference['latent_dimensions']} dimensiones → "
            f"2D mediante PCA. Las dos componentes muestran "
            f"{explained.sum():.1%} de la variación total."
        )

        with st.expander("¿Cómo interpretar este mapa?"):
            st.write(
                "La red convierte cada imagen en una lista de 512 activaciones. "
                "Imágenes que producen activaciones parecidas aparecen cerca "
                "entre sí. PCA comprime esas 512 coordenadas a dos para poder "
                "dibujarlas; por eso el mapa es una aproximación y puede "
                "contener regiones superpuestas."
            )

    except Exception as latent_exc:
        st.warning(
            "No fue posible construir el mapa. La clasificación continúa "
            "funcionando normalmente."
        )
        with st.expander("Detalle del problema"):
            st.code(str(latent_exc))


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

draw_column, map_column = st.columns([0.85, 1.65], gap="large")

with draw_column:
    st.subheader("Dibujo")
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

            with draw_column:
                result_left, result_right = st.columns([0.8, 1.2])

                with result_left:
                    st.subheader("Predicción")
                    st.markdown(
                        f"<div style='font-size:5rem; line-height:1; "
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
                    st.subheader("Opciones")
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
                        "La confianza es baja. Prueba dibujando el dígito más "
                        "grande y con un solo trazo continuo."
                    )

            with map_column:
                show_representation_space(model_input, prediction)
        else:
            with draw_column:
                st.caption("La predicción aparecerá automáticamente al dibujar.")
            with map_column:
                show_representation_space()

    except Exception as exc:
        with draw_column:
            st.error(f"Error al procesar la predicción: {exc}")
else:
    with draw_column:
        st.caption("La predicción aparecerá automáticamente al dibujar.")
    with map_column:
        show_representation_space()


with st.expander("Información técnica"):
    st.write(f"Modelo: `{MODEL_PATH.name}`")
    st.write(f"Forma de entrada: `{model.input_shape}`")
    st.write(f"Forma de salida: `{model.output_shape}`")
