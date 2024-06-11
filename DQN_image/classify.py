from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)

def winlose(image):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Crop image to fixed size
    x = 239
    y = 329
    w = 460
    h = 460
    image = image[y:y+h, x:x+w]

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    return index
