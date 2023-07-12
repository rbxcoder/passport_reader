from ultralytics import YOLO
import numpy
import cv2
import tensorflow as tf
import keras_ocr

# load a trained YOLOv8n model
model = YOLO("best.pt")  
pipeline = keras_ocr.pipeline.Pipeline()
names = model.names
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
                image_array = keras_ocr.tools.read(crop)
                predictions = pipeline.recognize([image_array])
                for prediction in predictions[0]:
                    text += prediction[0] 
                data[names[int(clsID)]]=text
    return data