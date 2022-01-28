from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os


app = Flask(__name__)

model = load_model('assets/model.h5')
class_names = ['card' ,'passport']


def predict_image(image_file):
    predict_img = image.load_img(
        image_file, 
        color_mode="rgb", 
        target_size=(124,124)
    )    
    array_image = image.img_to_array(predict_img)
    array_image = np.expand_dims(array_image, axis=0)
    prediction = model.predict(array_image)
    prediction_value = np.argmax(prediction)
    return prediction_value


@app.route("/classify", methods=['POST'])
def classify():
    file = request.files['image']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.root_path, "assets/images", filename))
    img = os.path.join(app.root_path, "assets/images", filename)
    value = predict_image(img)
    return jsonify({"prediction": class_names[value]})


@app.route("/classify-previews", methods=['POST'])
def classify_previews():
    json_data = request.get_json()
    image_name = json_data["image_name"]
    img = os.path.join(app.root_path, image_name)
    value = predict_image(img)
    return jsonify({"prediction": class_names[value]})


if __name__ == "__main__":
    app.run(debug=True)