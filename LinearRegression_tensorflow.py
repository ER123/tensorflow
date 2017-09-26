#Linear Regression

import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/code_python/AnacondaProjects/MNIST_data/",one_hot=True)

#设置参数
learning_rate = 0.001
training_step = 2000
display_step = 50

Xtrain = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Ytrain = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
nsamples = Xtrain.shape[0]

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

y_= tf.add(tf.multiply(X,W),b)
#tf.summary.histogram('y_',y_)

cross_entropy = tf.reduce_mean(tf.pow(y_-Y,2))/(2*nsamples)
#tf.summary.scalar('Loss Function', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_step):
        for (x,y) in zip(Xtrain, Ytrain):
            sess.run(train_step, feed_dict={X: x, Y: y})
    
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cross_entropy, feed_dict={X: Xtrain, Y: Ytrain})), "W=", sess.run(W), "b=", sess.run(b))
    print ("Optimization Finished!")
    print( "cost=", sess.run(cross_entropy, feed_dict={X: Xtrain, Y: Ytrain}), "W=", sess.run(W), "b=", sess.run(b))
