import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import base64
import io

with open('model/label.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

# model_path = 'model/zen-net-224.onnx'
model_path = 'model/mobilenetV2.onnx'
model = onnx.load(model_path)
session = ort.InferenceSession(model.SerializeToString())

def preprocess(img):
    # img = img / 255.
    img = cv2.resize(img, (128, 128))
    # img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def convert_base64(img_data):
    base64_decoded = base64.b64decode(img_data)

    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)
    return image_np

def predict(json_img):
    img = convert_base64(json_img)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    resp = labels[a[0]]
    prob = preds[a[0]]
    # print('class=%s ; probability=%f' %(resp,prob))

    return resp, prob
