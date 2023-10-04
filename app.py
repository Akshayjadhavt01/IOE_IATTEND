import cv2
import numpy as np
from keras.models import load_model
from PIL import ImageOps, Image
from keras.applications.mobilenet_v2 import preprocess_input
import time
import os
from dotenv import load_dotenv
import telebot

load_dotenv()

TOKEN = os.getenv('TOKEN')


bot = telebot.TeleBot(TOKEN)


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to a PIL Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Resize and preprocess the frame
    frame = ImageOps.fit(frame, (224, 224), Image.Resampling.LANCZOS)
    frame = np.asarray(frame)
    frame = preprocess_input(frame)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = frame

    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if(confidence_score > 0.96):
        bot.send_message(647289948, class_name[2:] + " is Present")

    #sending msg directly to bot from python
    # bot.send_message(647289948, class_name + " is Present")

    # Display the prediction and confidence score on the frame
    cv2.putText(frame, f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence Score: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    print(confidence_score)

    # time.sleep(1.0 / 5) # 5 fps

    # time.sleep(20) # 1 fps

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
