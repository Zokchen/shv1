import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('block_model(0.86%).h5')

# Function to predict if the image is a flower
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    
    class_names = ['Coffee_Station', 'Scholarship_Wall', 'alumni','brit','main gate','skill']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return predicted_class, confidence