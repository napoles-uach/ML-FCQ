import streamlit as st
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests
from scipy.ndimage.interpolation import zoom
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas
import os
#os.system("wget https://github.com/napoles-uach/ML-FCQ/raw/main/Demo/mi_modelo.h5")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

links={
  "bot":"https://assets8.lottiefiles.com/packages/lf20_g1pduE.json",
  "face" : "https://assets8.lottiefiles.com/packages/lf20_Sb1gLg.json",
  "process" : "https://assets8.lottiefiles.com/private_files/lf30_eTrSeS.json",
  "DS" : "https://assets7.lottiefiles.com/private_files/lf30_8z6ubjgk.json",
  "net":"https://assets1.lottiefiles.com/private_files/lf30_8npirptd.json",
  "bot-DS":"https://assets6.lottiefiles.com/temp/lf20_TOE9MF.json",
  "dash":"https://assets8.lottiefiles.com/packages/lf20_vpjqwyzx.json",
  "chem":"https://assets2.lottiefiles.com/private_files/lf30_k9hzIV.json"

}

st.title('Ejemplo Keras/MNIST')
col1,col2=st.columns([4,4])
with col1:
  st.image('https://miro.medium.com/max/1188/1*Ft2rLuO82eItlvJn5HOi9A.png')

with col2:
  st.image('https://miro.medium.com/max/1160/1*T-AyPeCdLDzHI0R952EmYg.gif')

def convim(im,nmax=28):
  lista1=[]
  for i in range(nmax):
    lista2=[]
    for j in range(nmax):
      bit=np.sum(im[i][j][:])
      lista2.append(bit)
    lista1.append(lista2)
  return lista1





(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

images_test = test_images

st.subheader('Definición de la arquitectura de la red neuronal :brain:')

code = '''model = keras.Sequential([
 layers.Dense(512, activation="relu"),
 layers.Dense(10, activation="softmax")
])'''

st.code(code, language='python')
n_neuronas=st.number_input('Opcional, da un valor para el numero de neuronas, default 512',min_value=10, max_value=1000, value=512)

#model = keras.Sequential([
# layers.Dense(n_neuronas, activation="relu"),
# layers.Dense(10, activation="softmax")
#])



st.subheader('Optimización usando backpropagation 🔙')
#model.compile(optimizer="rmsprop",
# loss="sparse_categorical_crossentropy",
# metrics=["accuracy"])
code_comp = '''model.compile(optimizer="rmsprop",
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"])
'''
st.code(code_comp,language='python')

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
#test_images[0].shape
#model.fit(train_images, train_labels, epochs=5, batch_size=128)
#@st.experimental_singleton
st.subheader('Fit: ajuste a los datos de entrenamiento 📈')
code_fit = 'model.fit(train_images, train_labels, epochs=5, batch_size=128)'
st.code(code_fit,language='python')
batch=st.select_slider('Opcional, da el valor del batch_size, default 128',[2**4,2**5,2**6,2**7,2**8],2**7)
def model_fit(model):
  history = model.fit(train_images, train_labels, epochs=5, batch_size=batch)
  return history


#train = st.sidebar.checkbox('train')

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#st.write(f"precision (accuracy) sobre el conjunto test: {test_acc}")

#if train:
#  with st.expander('plot'):
#    history = model_fit(model)
#    df = pd.DataFrame(history.history)
#    st.line_chart(df)
#st.stop()
os.system('ls')
model = keras.models.load_model('Demo/mi_modelo.h5')
st.markdown('''
#### Ahora que la red neuronal ha sido entrenada podemos ponerla a prueba entregandole algunas imagenes de numeros para ver que haga la prediccion correcta
''')
#tab1,tab2 = st.tabs(['tab1','tab2'])
with st.expander('Elige un numero'):

  test_digits = test_images[0:10]
  predictions = model.predict(test_digits)
  n=st.selectbox("Elige un indice para un numero de muestra",[0,1,2,3,4,5,6,7,8,9])
  fig, ax = plt.subplots()
  im = ax.imshow(images_test[n], cmap="binary")
  col1b,col2b = st.columns([4,4])
  with col1b:
    st.pyplot(fig)
  with col2b:
    st.write(images_test[n])

  st.header("Prediccion 🥁 ")

  col1c,col2c = st.columns([2,4])
  with col1c:
    st_lottie(load_lottieurl(links['bot']),key="1")
  with col2c:
    st.header('Yo digo que es un ')
    st.markdown('# '+str(predictions[n].argmax())+' ✨')


with st.expander('dibuja'):
  col1d,col2d = st.columns([2,2])
  with col2d:
    canvas_result = st_canvas(stroke_width= 10,height=28*5,width=28*5)
  
  if canvas_result.image_data is not None:
    ima =convim(canvas_result.image_data,28*5)
    grey = zoom(ima, 1/5)
    ima=np.array(grey)
    ima = ima.reshape( 28 * 28)
    ima = ima.astype("float32") / 255
    ima=[ima]
    ima=np.array(ima)
    pred_ima=model.predict(ima)
    with col1d:
      st_lottie(load_lottieurl(links['net']),key="2")
    col1d.subheader('🤔 Parece un '+str(pred_ima.argmax())+' ✨')
