# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""

from __future__ import division, print_function, absolute_import

import tflearn
import os
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17


from util import *
gdad = gdad_init()
X, Y = gdad.getXY()
testX, testY = gdad.getTestXY()
sample_size = gdad.sample_size


#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(60, 60))
print(X.shape)
# Building 'AlexNet'
network = input_data(shape=[None, 60, 60, 3])
network = conv_2d(network, 96, 6, strides=1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, sample_size, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

isTrain = False

model = tflearn.DNN(network, checkpoint_path='tmp/model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2,
                    best_checkpoint_path='bckp/bckp', best_val_accuracy=0.95) #define the best result store location

if(isTrain):
	# Training
	model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
	          show_metric=True, batch_size=64, snapshot_step=200,
	          snapshot_epoch=False, run_id='alexnet_oxflowers17')

	model.save("model_file")
else:
	model.load("bckp/model_file")

	print(Y[0])

	if(False):  #  show the image
	    imgssss = MatrixToImage(X[25])
	    gdad.show_img(imgssss)

	result = model.predict([X[0]])
	print(result)

	result = model.predict_label([X[0]])
	print(result,end="")