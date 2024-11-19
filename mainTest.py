import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')


image = cv2.imread('C:\\Users\\veerp\\OneDrive\\Desktop\\Tumor Detection\\Project\\Brain_Tumor_Classification-main\\pred\\pred7.jpg')

img = Image.fromarray(image, 'RGB')
img = img.resize((64, 64))  

img = np.array(img)
img = img / 255.0 

input_img = np.expand_dims(img, axis=0)

prediction = model.predict(input_img)
predicted_class = np.argmax(prediction, axis=1)[0] 

# output is in the form of 0 and 1 where 0 is for no tumor and 1 is for tumor
print(predicted_class)
