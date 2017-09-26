from __future__ import print_function

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/code_python/MNIST_data/", one_hot=True)

#Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1
logdir_mnist = '/tmp/mnist_logs/Adam'

#Network Parameters
n_classes = 10 
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256

X = tf.placeholder(tf.float32, [None, n_input], name='Input_data')
Y = tf.placeholder(tf.float32, [None, n_classes], name='Labels')

weigths = {
	'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='w1'),
	'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='w2'),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='out')
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
	'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
	'out': tf.Variable(tf.random_normal([n_classes]), name='out')
}

def multiplayer_perceptron(x):
	layer1 = tf.add(tf.matmul(x, weigths['w1']), biases['b1'])
	layer1 = tf.nn.relu(layer1)
	tf.summary.histogram("Layer1", layer1)
	layer2 = tf.add(tf.matmul(layer1, weigths['w2']), biases['b2'])
	layer2 = tf.nn.relu(layer2)
	tf.summary.histogram("Layer2", layer2)
	out_layer = tf.matmul(layer2, weigths['out']) + biases['out']
	return out_layer

with tf.name_scope('Model'):
	logits = multiplayer_perceptron(X)

with tf.name_scope('Loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

with tf.name_scope('Adma'):
	optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#with tf.name_scope('SGD'):
#	SDGoptimize = tf.train.GradientDescentOptimizer(learning_rate)
#	grads = tf.gradients(loss, tf.trainable_variables())
#	grads = list(zip(grads, tf.trainable_variables()))
#	apply_grads = SDGoptimize.apply_gradients(grads)

with tf.name_scope('Accuracy'):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)

for var in tf.trainable_variables():
	tf.summary.histogram(var.name, var)

#for grad, var in grads:
#	tf.summary.histogram(var.name+'/gradients', grad)

merged_summary = tf.summary.merge_all()

with tf.Session() as sess:

	sess.run(init)

	summary_writer = tf.summary.FileWriter(logdir_mnist, graph=tf.get_default_graph())

	for epoch in range(training_epochs):
		avg_cost = 0.
		totalBatch = int(mnist.train.num_examples/batch_size)
		for i in range(totalBatch):
			batch = mnist.train.next_batch(batch_size)
			_, c, summary = sess.run([optimize, loss, merged_summary], feed_dict={X: batch[0], Y: batch[1]})
			summary_writer.add_summary(summary, epoch * totalBatch + i)
			avg_cost += c / totalBatch
		if (epoch + 1) % display_step == 0:
			print("Epoch=", '%04d' % (epoch + 1), "cost=", '{:.04f}'.format(avg_cost))
	print("Optimize finished!")
	print("Accuracy=", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

