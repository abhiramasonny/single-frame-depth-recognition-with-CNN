from flask import Flask, render_template, request, Response
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_blob = request.get_data()
    
    # Convert blob data to OpenCV format
    nparr = np.frombuffer(frame_blob, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Dummy AI processing: Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale frame back to blob data
    retval, buffer = cv2.imencode('.jpg', gray_frame)
    processed_blob = buffer.tobytes()
    
    return Response(processed_blob, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
