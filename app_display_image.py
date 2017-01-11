import os
import tensorflow as tf
import sys

from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        if upload and allowed_file(upload.filename):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = secure_filename(upload.filename)
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
        else:
            return render_template("not_ok.html")

    ####TF####
    # change this as you see fit
    image_path = '/feridun/images/'+filename

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("/feridun/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("/feridun/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        result = {'score':0, 'name':''}
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > result['score']:
                result['score']=score
                result['name']=human_string
    ##TF ENDS##

    return render_template("complete_display_image.html", image_name=filename, result=result['name'], score=result['score'])

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
