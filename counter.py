import jetson.inference
import jetson.utils
import numpy as np
import cv2
import os
from time import time

# Parameters
PERSON_CLASS = 1
DETECTION_THRESHOLD = 0.5



class Counter:

    def __init__(self):

        self.load_model()


    def load_model(self):
        # Model loading
        print('[AI] Loading network ...')
        self.net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.3)
        print('[AI] Loading network OK')



    async def load_image_from_request(self, file):
        img_data = await file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert colors
        imageRGB = cv2.cvtColor(img_np , cv2.COLOR_BGR2RGB)
        return jetson.utils.cudaFromNumpy(imageRGB)

    def predict(self, img):


        inference_start_time = time()

        detections = self.net.Detect(img, img.width,img.height)

        inference_time = time() - inference_start_time

        filtered = [x for x in detections if x.ClassID  == 1]


        count = len(filtered)


        return {
        'prediction': count,
        'inference_time': inference_time,
        }
