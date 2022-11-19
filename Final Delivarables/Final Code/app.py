from flask import Flask,render_template
# Flask-It is our framework which we are going to use to r #request-for accessing file which was uploaded by the user #import operator
import cv2 # opencv library
from tensorflow.python.keras.models import load_model#to load our
import numpy as np
#import os 

app = Flask(__name__, template_folder='static')
model = load_model('disaster.h5')
print("load_model from disk")


@app.route('/')
def home():
    return render_template('home.html', title='Home', active_page='home')


@app.route('/intro')
def intro():
    return render_template('intro.html', title='Home', active_page='introduction')


@app.route('/upload')
def upload():
    return render_template('upload.html', title='Home', active_page='upload')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    
    cap = cv2.VideoCapture('uploads/fire.mp4')
    while True:
        _, frame = cap.read()
    
        frame = cv2.flip(frame, 1)
        while True:
            (grabbed, frame) = cap.read()
            W,H= None, None
            if not grabbed:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            x = np.expand_dims(frame, axis=0)
            result = np.argmax(model.predict(x), axis=-1)
            index = ['Cyclone', 'Earthquake', 'Flood', 'Wildefire']
            result = str(index[result[0]])
            cv2.putText(output, "activity: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            cv2.imshow("output", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        print("[INFO] cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        return render_template("upload.html")


if __name__ == "__main__":
   # running the app
    app.run(debug=False)