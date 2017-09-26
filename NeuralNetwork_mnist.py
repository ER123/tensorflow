from __future__ import print_function

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/code_python/MNIST_data/", one_hot=True)

#Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

#Network Parameter
n_hidden1 = 256
n_hidden2 = 256
n_input = 784
n_classes = 10

def nerual_nt(x_dict):
	x = x_dict['images']
	layer1 = tf.layers.dense(x, n_hidden1)
	layer2 = tf.layers.dense(layer1, n_hidden2)
	out_layer = tf.layers.dense(layer2, n_classes)
	return out_layer

def model_fn(features, labels, mode):
	logits = nerual_nt(features)

	pred_classes = tf.argmax(logits, axis=1)
	pred_prob = tf.nn.softmax(logits)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int32)))

	optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

	accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

	estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss, train_op=optimize, eval_metric_ops={'accuracy':accuracy})

	return estim_specs

model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=batch_size, num_epochs=None, shuffle=False)

model.train(input_fn, steps=num_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn)

print("Test Accuracy", e['accuracy'])