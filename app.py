import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from detection import *
# from preprocessing import *

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
imagepath='getImage.png'
@app.route('/api/passportdetection', methods=['GET','POST'])
def detect():
    try:
        imageb = request.get_json()['image']
    except:
         return jsonify({'message':'no image sent!'})
    try:
        with open(imagepath, "wb") as fh:
            fh.write(base64.b64decode(str(imageb)))
    except:
        return jsonify({'message':'error image format!'})
    imageDet=cv2.imread('getImage.png')
    data = detection(imagepath,imageDet)
    return jsonify(data)

            

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


