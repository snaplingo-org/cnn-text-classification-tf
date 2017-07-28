#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from flask import Flask, jsonify, render_template, request
import json

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1501126437/checkpoints/",
                       "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(
            "output/predictions").outputs[0]

        def evalFunc(input):
            print('input',input)
            x_raw = input
            x_test = np.array(list(vocab_processor.transform(x_raw)))
            pred = sess.run(
                predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            return pred.tolist()

# webapp
app = Flask(__name__)

@app.route('/api/checkFeedback', methods=['POST'])
def cnn():
    data = request.get_data()
    data.decode("utf-8")
    ls = json.loads(data)
    texts = [obj.get('text') for obj in ls]
    ids = [obj.get('id') for obj in ls]
    print(ids)
    output = evalFunc(texts)
    print(output)
    results = dict(zip(ids, output))
    print(results)
    return jsonify(results=results)
if __name__ == '__main__':
    app.run(host='0.0.0.0')
