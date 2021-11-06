# Person counter
A wrapper for the ssd_mobilenet_v2_coco_2018_03_29 object detection model where only the number of people in a frame is returned.
This wrapper allows the model to be interacted with using HTTP calls, making it well suited for microservice applications.

## API
| Route | Method | query/body | Description |
| --- | --- | --- | --- |
| / | GET | - | Show application configuration |
| /predict | POST | multipart-from data with images as files | Count the number of people in each picture |
