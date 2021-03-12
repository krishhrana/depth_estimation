from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import urllib

app = Flask(__name__)

export_dir = 'SavedModel/tf_savedmodel/1'
global graph
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
tf.saved_model.loader.load(sess, ["serve"], export_dir)


@app.route("/predict", methods=["POST"])
def getDepth():
    with graph.as_default():
        x = graph.get_tensor_by_name("im0:0")
        model = graph.get_tensor_by_name("truediv:0")
        image_file = request.json['image']
        image = np.array(image_file)
        image = image / 255
        image = np.array(image).reshape(1, 320, 640, 3)
        disp = sess.run(model, feed_dict={x: image})
    return jsonify({'prediction': disp.tolist()})


def get_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


if __name__ == "__main__":
    app.run(debug=True)
