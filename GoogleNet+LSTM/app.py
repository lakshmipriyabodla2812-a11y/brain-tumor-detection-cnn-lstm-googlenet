from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = 'the random string'


def get_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/reg")
def reg():
    return render_template("ureg.html")


@app.route('/regback', methods=["POST"])
def regback():
    flash("Registration successful (Demo Mode)", "success")
    return render_template('user.html')


@app.route('/userlog', methods=['POST', 'GET'])
def userlog():
    if request.method == "POST":
        flash("Welcome to the system!", "success")
        return redirect(url_for('upload1'))
    return render_template('user.html')


@app.route("/userhome")
def userhome():
    return render_template("userhome.html")


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':

        myfile = request.files['file']
        fn = myfile.filename

        upload_folder = 'images'
        os.makedirs(upload_folder, exist_ok=True)

        upload_path = os.path.join(upload_folder, fn)
        myfile.save(upload_path)

        classes = ['No Tumor', 'Tumor']
        target_size = (299, 299)

        # Preprocess
        test_image = image.load_img(upload_path, target_size=target_size)
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Load model
        model_path = 'alg/inception_lstm_model01.h5'
        model = load_model(model_path)

        # Prediction
        result = model.predict(test_image)
        prediction = classes[np.argmax(result)]

        msg = ""
        if prediction == 'Tumor':
            msg = "Based on the test results, a tumor is detected. Please consult a doctor."

        img = cv2.imread(upload_path)
        img = cv2.resize(img, (299, 299))

        output_filename = "result_" + fn
        output_path = os.path.join(upload_folder, output_filename)

        try:
            # Only run Grad-CAM if tumor detected
            if prediction == 'Tumor':

                # Auto-detect last conv layer
                last_conv_layer = None
                for layer in reversed(model.layers):
                    if 'conv' in layer.name:
                        last_conv_layer = layer.name
                        break

                if last_conv_layer is None:
                    raise Exception("No conv layer found")

                heatmap = get_gradcam(model, test_image, last_conv_layer)

                heatmap = cv2.resize(heatmap, (299, 299))
                heatmap = np.uint8(255 * heatmap)

                # Heatmap overlay
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

                # Threshold for strong region
                _, thresh = cv2.threshold(heatmap, 180, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

                if valid_contours:
                    largest = max(valid_contours, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(largest)

                    center = (int(x), int(y))
                    radius = int(radius)

                    cv2.circle(overlay, center, radius, (0, 0, 255), 3)

                cv2.imwrite(output_path, overlay)

            else:
                # No tumor → show clean image
                cv2.imwrite(output_path, img)

        except Exception as e:
            print("Grad-CAM error:", e)
            cv2.imwrite(output_path, img)

        return render_template("template.html",
                               image_name=output_filename,
                               text=prediction,
                               msg=msg)

    return "Please upload an image."


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/upload1')
def upload1():
    return render_template("upload.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)