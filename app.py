import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')
 
def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"  
    elif class_no == 1:
        return "Yes Brain Tumor"  

def get_result(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64)) 
    image = np.array(image) / 255.0  
    input_img = np.expand_dims(image, axis=0)  

    
    prediction = model.predict(input_img)

   
    predicted_class = np.argmax(prediction, axis=1)[0]
    return get_class_name(predicted_class)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'pred', secure_filename(f.filename))  
        f.save(file_path)

        
        result = get_result(file_path)

       
        return str(result)

    return None


if __name__ == '__main__':
    app.run(debug=True)
