# flask_tensorflow/app.py

from flask import Flask, jsonify, request, render_template
from scipy.misc import imread, imresize
import numpy as np
import time
import tensorflow as tf

app = Flask(__name__)
# MODEL_PATH = '/usr/src/app/model/graph.pb'
# LABEL_PATH = '/usr/src/app/model/labels.txt'

MODEL_PATH = './model/graph.pb'
LABEL_PATH = './model/labels.txt'
##################################################
# REST API Endpoints For Web App
##################################################

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/classification')
def classification():
    file_path = request.args['file_path']
    # Load in an image to classify and preprocess it
    image_data = imread(file_path)
    image = imresize(image_data, [224, 224])
    x_input = np.expand_dims(image, axis=0)

    # Get the predictions (output of the softmax) for this image
    t = time.time()
    tf_results = sess.run(output_tensor, {input_tensor : x_input})
    dt = time.time() - t
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))

    # Single image in this batch
    predictions = tf_results[0]

    # The probabilities should sum to 1
    assert np.isclose(np.sum(predictions), 1)

    class_label = np.argmax(predictions)
    app.logger.info("Image %s classified as %d" % (file_path, class_label))

    return jsonify(apiResponseCreator(classification_labels, predictions.tolist()))

##################################################
# Utilities
##################################################

def readLabels():
    # Read each line of label file and strip \n
    labels = [label.rstrip('\n') for label in open(LABEL_PATH)]
    return labels

def apiResponseCreator(labels, classifications):
    return dict(zip(labels, classifications))

def printTensors(model_file):
    # read protobuf into graph_def
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for operation in graph.get_operations():
        print(operation.name)

##################################################
# Main..
##################################################

if __name__ == '__main__':
    print('Starting TensorFlow Server')
    print('Loading Model...')
    # Read the graph definition file
    with open(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Load the graph stored in `graph_def` into `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    # Enforce that no new nodes are added
    graph.finalize()
    print('Done.')

    print('Configuring TensorFlow Graph..')
    # Create the session that we'll use to execute the model
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement = True
    )

    sess = tf.Session(graph=graph, config=sess_config)
    print('Done.')
    # Get the input and output operations
    # If you don't know the operation name use this:
    # printTensors('./model/graph.pb')
    input_op = graph.get_operation_by_name('input')
    input_tensor = input_op.outputs[0]
    output_op = graph.get_operation_by_name('final_result')
    output_tensor = output_op.outputs[0]
    # Get class labels
    classification_labels = readLabels()

    app.run(debug=True, host='0.0.0.0')
