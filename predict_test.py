from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("head_position_model.h5")
class_names = ['headdownframes', 'headupframes']

img_path = "head_up_random_test_3.jpg"  
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0  

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]



print(f"Prediction: {predicted_class}")
