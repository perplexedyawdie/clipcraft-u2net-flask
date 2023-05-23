from io import BytesIO
from flask import Flask, request, send_file, Response
import base64
import cv2
import numpy as np
import onnxruntime
import os
from keras.preprocessing.image import image_utils
from PIL import Image as Img

app = Flask(__name__)


class u2net():
    def __init__(self):
        try:
            cvnet = cv2.dnn.readNet('u2net.onnx')
        except:
            print('opencv read onnx failed!!!')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession('u2net.onnx', so)
        self.input_size = 320
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name

    def detect(self, srcimg):        
        img = cv2.resize(srcimg, dsize=(self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - self.mean) / self.std
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run(["1799"], {"input.1": blob})
        result = np.array(outs[0]).squeeze()
        result = (1 - result)
        min_value = np.min(result)
        max_value = np.max(result)
        result = (result - min_value) / (max_value - min_value)
        result *= 255
        return result.astype('uint8')
mynet = u2net()   

@app.route('/', methods=["POST"])
def index():
    file = request.files['file']
    # Read the file contents
    file_content = file.read()
    # Create an in-memory file-like object
    file_obj = BytesIO()
    # Create an in-memory buffer using BytesIO
    buffer = BytesIO(file_content)
    srcimg = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    # Detect the object using U2net
    result = mynet.detect(srcimg)

    # Resize the result to the original image size
    result = cv2.resize(result, (srcimg.shape[1], srcimg.shape[0]))

    _, buffer = cv2.imencode('.png', result)
    image_data = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    out_img = image_utils.img_to_array(image_data)
    out_img /= 255  # Normalization to [0,1]

    # Define threshold to convert U2net output into binary mask
    THRESHOLD = 0.9
    # Create mask from out_img (values over the threshold are set to 1 and those under it are set to 0)
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0

    # Invert the binary mask
    out_img = 1 - out_img

    # Convert mask to 4-channel image, with mask as alpha (transparency) channel
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init

    # Create three-channel image from out_img
    out_img_rgb = np.repeat(a_layer, 3, axis=-1)

    # Append the alpha layer to get a RGBA image
    rgba_out = np.concatenate([out_img_rgb, a_layer], axis=2)

    # Load the original image
    input = image_utils.load_img(BytesIO(file_content))
    inp_img = image_utils.img_to_array(input)
    inp_img /= 255  # Normalization to [0,1]

    # Add an alpha channel to original image
    rgba_inp = np.concatenate([inp_img, a_layer], axis=2)

    # Remove background
    rem_back = rgba_inp * rgba_out

    # Create a PIL Image from rem_back
    rem_back_image = Img.fromarray((rem_back * 255).astype('uint8'), 'RGBA')

    # Create an in-memory buffer for the image
    image_buffer = BytesIO()
    rem_back_image.save(image_buffer, format='PNG')
    image_buffer.seek(0)

    # Send the rem_back image as a file blob
    return send_file(image_buffer, mimetype='image/png')

