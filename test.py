import base64
import json

import cv2
import numpy as np
import requests

#select one image that want to classify
# img = "0_incorrect_mask.jpg"
# img = "1_using_mask.jpg"
img = "2_without_mask.jpg"

with open(img, "rb") as img_file:
    img_string = base64.b64encode(img_file.read())

img_string = img_string.decode("utf-8")
payload = {
    "image": img_string,
}

response = requests.post("http://127.0.0.1:5000/classify_mask", headers={"Content-Type":"application/json"}, data=json.dumps(payload))
response_dict = response.text

print(response_dict)