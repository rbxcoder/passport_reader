from ultralytics import YOLO
import numpy
import cv2
import tensorflow as tf
import easyocr


# load a trained YOLOv8n model
model = YOLO("best.pt")  
# pipeline = keras_ocr.pipeline.Pipeline()
names = model.names
reader = easyocr.Reader(['en'])
scale_percent = 150 # percent of original size
def detection(imagepath,imageDet) :
    data={}
    
    # predict on an image
    detection_output = model.predict(source=imagepath, conf=0.40, save=False) 

    # Display numpy array
    DP = detection_output[0].numpy()
    if len(DP) != 0:
            for i in range(len(detection_output[0])):
                text=""
                boxes = detection_output[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0].astype(int)
                crop = imageDet[bb[1]:bb[3], bb[0]:bb[2]]
                width = int(crop.shape[1] * scale_percent / 100)
                height = int(crop.shape[0] * scale_percent / 100)
                dim = (width, height)
                crop = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
                results = reader.readtext(crop)
                detected_text = [result[1] for result in results]
                # Join the detected text into a single string
                final_text = ' '.join(detected_text)
                # Close the reader
                data[names[int(clsID)]]=final_text
    return data