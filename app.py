import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('notebooks/traffic_classifier.h5')
    return model

classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

st.title("🚦 Traffic Sign Recognition System")
st.write("Upload an image of a traffic sign, and the Deep Learning model will identify it.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify Traffic Sign"):
        with st.spinner('Analyzing...'):
            try:
                model = load_model()
                
                img = image.resize((64, 64)) 
                
                img_array = np.array(img)
                
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]
                
                img_array = np.expand_dims(img_array, axis=0)
                
                img_array = img_array / 255.0
                
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100
                
                label = classes[predicted_class]
                
                st.success(f"Prediction: **{label}**")
                st.info(f"Confidence: {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"Error processing image: {e}")