from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Load the model
model = load_model("action.h5", compile=False)

def choose(image):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(np.array(image), (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    return index
