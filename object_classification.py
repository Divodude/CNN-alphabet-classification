import cv2
import numpy as np
from tensorflow.keras.models import load_model

l1 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
model = load_model(r"path of model")
captur = cv2.VideoCapture(1)
l1 = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()

img=cv2.imread(r"path of image")
img= cv2.resize(img,(64,64)) 
img= np.expand_dims(img/255, axis=0)
prediction = model.predict(img)
idx = np.argmax(prediction)
if prediction[0,idx] > 0.4:
        predicted_alphabet=l1[idx]
        print(predicted_alphabet)





        
