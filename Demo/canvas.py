import streamlit as st
import tensorflow
from tensorflow import keras
import numpy as np
from scipy.ndimage.interpolation import zoom
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = keras.models.load_model('Demo/mi_modelo.h5')

def convim(im,nmax=28):
  lista1=[]
  for i in range(nmax):
    lista2=[]
    for j in range(nmax):
      bit=np.sum(im[i][j][:])
      lista2.append(bit)
    lista1.append(lista2)
  return lista1

with st.expander('Draw a digit:'):
  # Display canvas for drawing
  canvas_result = st_canvas(stroke_width=10, height=28*5, width=28*5)
  
  # Process drawn image and make prediction using model
  if canvas_result.image_data is not None:
    ima = convim(canvas_result.image_data, 28*5)
    grey = zoom(ima, 1/5)
    ima = np.array(grey)
    ima = ima.reshape(28 * 28)
    ima = ima.astype("float32") / 255
    ima = [ima]
    ima = np.array(ima)
    pred_ima = model.predict(ima)
    st.header('Prediction:')
    st.write('This image appears to be a ' + str(pred_ima.argmax()) + '.')
