import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/code_python/AnacondaProjects/MNIST_data/", one_hot=True)

#设置参数
learning_rate = 0.001
training_epoch = 20
batch_size = 50
display_epoch = 1

piexls = 784
n_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

W = tf.Variable(tf.zeros([784, 10]), name="weight")
b = tf.Variable(tf.zeros([10]), name="bias")

y = tf.nn.softmax(tf.matmul(x,W) + b)
tf.summary.histogram('y', y)

cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=1))
tf.summary.scalar('loss', cost)

optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merge_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)
	for epoch in range(training_epoch):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			feeds = {x: batch_xs, y_: batch_ys}
			_, c = sess.run([optimize, cost], feed_dict=feeds)
			avg_cost += c/total_batch
			summary_str = sess.run(merge_summary,feed_dict={x: batch_xs, y_: batch_ys})
			summary_writer.add_summary(summary_str, epoch)
		if (epoch+1) % display_epoch == 0:
			print("Epoch=",'%04d'%(epoch+1), "cost=",'{:.5f}'.format(avg_cost))

	print("Optimizer finished!")
	# Test model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))	
