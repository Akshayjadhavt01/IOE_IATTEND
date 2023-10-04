import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import ImageOps, Image
import time
import os
from dotenv import load_dotenv
import telebot

load_dotenv()

#TOKEN = os.getenv('TOKEN')
TOKEN = "6422606819:AAFVIE5DKZ5x2xRtzoXr7ALztbZgFfy-rM8"
bot = telebot.TeleBot(TOKEN)

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the input details from the TensorFlow Lite model
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to a PIL Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Resize and preprocess the frame
    frame = ImageOps.fit(frame, (input_shape[1], input_shape[2]), Image.LANCZOS)
    frame = frame.convert("L")  # Convert to grayscale
    frame = frame.resize((input_shape[1], input_shape[2]), Image.LANCZOS)
    frame = np.asarray(frame, dtype=np.uint8)  # Convert to UINT8 data type

    # Convert the grayscale frame to 3 channels (RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Expand the dimensions to match the expected 4D input shape
    frame = np.expand_dims(frame, axis=0)

    # Set the input tensor of the TensorFlow Lite model
    interpreter.set_tensor(input_details[0]['index'], frame)

    # Run inference
    interpreter.invoke()

    # Get the output details and predictions
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(output_data)
    class_name = class_names[index]
    confidence_score = output_data[0][index]

    if confidence_score > 240:
        print("It's "+class_name[2:])
        bot.send_message(647289948, class_name[2:] + " is Present")
    elif confidence_score >= 100 <= 240:
        print("It looks like "+class_name[2:])
    else:
        print("Cant Detect")

    # Display the prediction and confidence score on the frame
    cv2.putText(frame[0], f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame[0], f"Confidence Score: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam Feed", frame[0])

    # print(confidence_score)

    # time.sleep(1.0 / 5) # 5 fps

    # time.sleep(10) # 1 fps

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

