import tensorflow as tf
import numpy as np
import cv2
import os
from time import time

# Parameters
PERSON_CLASS = 1
DETECTION_THRESHOLD = 0.5

# Model download
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_PATH = f'{os.getcwd()}/models/{MODEL_NAME}/saved_model'


class Counter:

    def __init__(self):

        self.load_model()

    def load_model(self):
        # Model loading
        print('[AI] Loading model ...')
        model_temp = tf.compat.v2.saved_model.load(MODEL_PATH, None)
        self.model = model_temp.signatures['serving_default']
        print('[AI] Loading model OK')



    async def load_image_from_request(self, file):
         img_data = await file.read()
         nparr = np.frombuffer(img_data, np.uint8)
         return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    async def predict(self, file):


        inference_start_time = time()
        img = await self.load_image_from_request(file)
        img = np.expand_dims(img, axis=0)

        model_input = tf.convert_to_tensor(img, dtype=tf.uint8)
        output_dict = self.model(model_input)

        results_classes_numpy = output_dict['detection_classes'].numpy()
        results_scores_numpy = output_dict['detection_scores'].numpy()

        counts = []

        # Not very pretty
        for current_classes, current_scores in zip(results_classes_numpy, results_scores_numpy):

            classes = [int(x) for x in current_classes.tolist()]
            scores = current_scores.tolist()

            person_count_for_current_frame = 0

            for i, object_class in enumerate(classes):
                if  object_class == PERSON_CLASS and scores[i] > DETECTION_THRESHOLD:

                    # Count is done here
                    person_count_for_current_frame = person_count_for_current_frame + 1

            counts.append(person_count_for_current_frame)

        inference_time = time() - inference_start_time

        print('Response')

        return {
        'prediction': counts,
        'inference_time': inference_time,
        }
