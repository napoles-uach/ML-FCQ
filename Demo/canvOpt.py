import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import zoom
from streamlit_drawable_canvas import st_canvas
st.markdown("# Digit :blue[Recognition] :green[App] :pencil: 💻🤖")

# Load trained model
model = tf.keras.models.load_model('Demo/mi_modelo.h5')

def process_image(image_data, size=28):
  """Convert drawn image to grayscale and resize to 28x28."""
  # Convert image to grayscale
  grayscale_image = np.sum(image_data, axis=2)
  # Resize image
  resized_image = zoom(grayscale_image, size / grayscale_image.shape[0])
  # Normalize pixel values
  normalized_image = resized_image.astype(np.float32) / 255
  # Return image as a single row
  return normalized_image.reshape(1, -1)

st.write('Draw a digit:')
# Display canvas for drawing
canvas_result = st_canvas(stroke_width=10, height=28*5, width=28*5)
  
# Process drawn image and make prediction using model
if np.any(canvas_result.image_data):
    #st.write(canvas_result.image_data)
    # Convert drawn image to grayscale and resize to 28x28
    processed_image = process_image(canvas_result.image_data)
    # Make prediction using model
    prediction = model.predict(processed_image).argmax()
    # Display prediction
    st.header('Prediction:')
    st.markdown('This number appears to be a \n # :red[' + str(prediction) + ']')
else:
    # Display message if canvas is empty
    st.header('Prediction:')
    st.write('No number drawn, please draw a digit to get a prediction.')
