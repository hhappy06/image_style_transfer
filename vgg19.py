import os
import numpy as np
import tensorflow as tf

_VGG19_IMAGE_MEAN = [103.939, 116.779, 123.68]
_WEIGHT_INDEX = 0
_BIAS_INDEX = 1
_REGULAR_FACTOR = 1.0e-4
_LEARNING_RATE = 1.0e-4

_VGG19_NETWORK = {
	'conv1_1': [3, 64],
	'conv1_2': [3, 64],
	'conv2_1': [3, 128],
	'conv2_2': [3, 128],
	'conv3_1': [3, 256],
	'conv3_2': [3, 256],
	'conv3_3': [3, 256],
	'conv3_4': [3, 256],
	'conv4_1': [3, 512],
	'conv4_2': [3, 512],
	'conv4_3': [3, 512],
	'conv4_4': [3, 512],
	'conv5_1': [3, 512],
	'conv5_2': [3, 512],
	'conv5_3': [3, 512],
	'conv5_4': [3, 512],
	'fc6': [4096],
	'fc7': [4096],
	'fc8': [1000],
}

_CONV_KERNEL_STRIDES = [1, 1, 1, 1]
_MAX_POOL_KSIZE = [1, 2, 2, 1]
_MAX_POOL_STRIDES = [1, 2, 2, 1]

class VGG19:
	def __init__(self, initialized_parameter_file):
		if not initialized_parameter_file and not os.path.exists(initialized_parameter_file):
			print("initialized_parameter_file is None or the file does not exist")
			return
		
		self.initialized_parameter_dict = np.load(initialized_parameter_file, encoding = 'latin1').item()

	def inference(self, input_tensor):
		# input_images is a placeholder with [None, height, width, nchannels]
		r, g, b = tf.split(input_tensor,3, 3)
		whiten_images = tf.concat([
			b - _VGG19_IMAGE_MEAN[0],
			g - _VGG19_IMAGE_MEAN[1],
			r - _VGG19_IMAGE_MEAN[2]], 3)

		net = {}
		# construct VGG19 network -- convolution layer
		net['conv1_1'] = self._construct_conv_layer(whiten_images, 'conv1_1')
		net['conv1_2'] = self._construct_conv_layer(net['conv1_1'], 'conv1_2')
		net['pool1'] = self._max_pool(net['conv1_2'], 'pool1')

		net['conv2_1'] = self._construct_conv_layer(net['pool1'], 'conv2_1')
		net['conv2_2'] = self._construct_conv_layer(net['conv2_1'], 'conv2_2')
		net['pool2'] = self._max_pool(net['conv2_2'], 'pool2')

		net['conv3_1'] = self._construct_conv_layer(net['pool2'], 'conv3_1')
		net['conv3_2'] = self._construct_conv_layer(net['conv3_1'], 'conv3_2')
		net['conv3_3'] = self._construct_conv_layer(net['conv3_2'], 'conv3_3')
		net['conv3_4'] = self._construct_conv_layer(net['conv3_3'], 'conv3_4')
		net['pool3'] = self._max_pool(net['conv3_4'], 'pool3')

		net['conv4_1'] = self._construct_conv_layer(net['pool3'], 'conv4_1')
		net['conv4_2'] = self._construct_conv_layer(net['conv4_1'], 'conv4_2')
		net['conv4_3'] = self._construct_conv_layer(net['conv4_2'], 'conv4_3')
		net['conv4_4'] = self._construct_conv_layer(net['conv4_3'], 'conv4_4')
		net['pool4'] = self._max_pool(net['conv4_4'], 'pool4')

		net['conv5_1'] = self._construct_conv_layer(net['pool4'], 'conv5_1')
		net['conv5_2'] = self._construct_conv_layer(net['conv5_1'], 'conv5_2')
		net['conv5_3'] = self._construct_conv_layer(net['conv5_2'], 'conv5_3')
		net['conv5_4'] = self._construct_conv_layer(net['conv5_3'], 'conv5_4')

		return net

	def _construct_conv_layer(self, input_layer, layer_name):
		assert layer_name in _VGG19_NETWORK
		weight = tf.constant(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX], name= layer_name + "filter")
		bias = tf.constant(self.initialized_parameter_dict[layer_name][_BIAS_INDEX], name= layer_name + "biases")
		conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES, padding = 'SAME')
		active = tf.nn.relu(tf.nn.bias_add(conv, bias))
		return active

	def _construct_full_connection_layer(self, input_layer, layer_name, active = True):
		assert layer_name in _VGG19_NETWORK
		weight = tf.constant(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX], name= layer_name + "filter")
		bias = tf.constant(self.initialized_parameter_dict[layer_name][_BIAS_INDEX], name= layer_name + "biases")

		input_dimension = 1
		for dim in input_layer.get_shape().as_list()[1:]:
			input_dimension *= dim
		reshape_input = tf.reshape(input_layer, [-1, input_dimension])

		if active:
			return tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape_input, weight), bias))
		return tf.nn.bias_add(tf.matmul(reshape_input, weight), bias)

	def _max_pool(self, input_layer, name):
		return tf.nn.max_pool(input_layer, ksize = _MAX_POOL_KSIZE, strides = _MAX_POOL_STRIDES, padding = 'SAME', name = name)