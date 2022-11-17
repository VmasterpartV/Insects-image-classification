import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  print(array)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  print(answer)
  if answer == 0:
    print("pred: chinche marrón")
  elif answer == 1:
    print("pred: chinche verde")
  elif answer == 2:
    print("pred: gusano de soya")

  return answer

predict('./prueba/prueba24.jpg')
