import numpy as np
import os
import tensorflow as tf
from vgg19 import VGG19
import scipy.misc

_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
_CONTENT_LAYER = ['conv4_2']

_LEARNING_RATE = 1.0E-3
_OUTPUT_INFO_FREQUENCE = 10
_OUTPUT_RESULT_FREQUENCE = 100

class StyleTransfer():
	def __init__(self, vgg19_path):
		if not os.path.exists(vgg19_path):
			print 'the vgg19 path is wrong'
			return

		self.vgg19 = VGG19(vgg19_path)

	def image_style_transfer(self, image_content, image_style, content_weight, style_weight, tv_weight, max_iteration = 1000, init_image = None):
		image_shape = image_content.shape
		with tf.Graph().as_default(), tf.Session() as session:
			# extract feature from content and style
			print 'constructing feature extractor net...'
			feature_image_tensor = tf.placeholder(tf.float32, shape = [1, image_shape[0],image_shape[1],image_shape[2]])
			feature_net = self.vgg19.inference(feature_image_tensor)

			print 'extracting style feature from style image...'
			feature_extractor = [feature_net[layer] for layer in _STYLE_LAYERS]
			features = session.run(feature_extractor, feed_dict = {
					feature_image_tensor: [image_style]
					})
			style_feature_dic = {}
			for idx, layer in enumerate(_STYLE_LAYERS):
				feature = features[idx]
				feature = np.reshape(feature, (-1, feature.shape[-1]))
				gram = np.matmul(feature.T, feature) / feature.size
				style_feature_dic[layer] = gram

			print 'extracting content feature from content image...'
			content_feature_dic = {}
			feature_extractor = [feature_net[layer] for layer in _CONTENT_LAYER]
			features = session.run(feature_extractor, feed_dict = {
					feature_image_tensor: [image_content]
					})
			content_feature_dic = {}
			for idx, layer in enumerate(_CONTENT_LAYER):
				content_feature_dic[layer] = features[idx]

			# predict new image
			print 'constructing image-style-transfer net...'
			init_tensor_value = (content_weight * image_content + style_weight * image_style) / (content_weight + style_weight)
			if init_image is not None:
				init_tensor_value = init_image
			init_tensor_value = init_tensor_value.flatten()

			pred_image = tf.get_variable(
				name = 'varialbe',
				shape = [1, image_shape[0],image_shape[1],image_shape[2]],
				initializer = tf.constant_initializer(init_tensor_value))

			pred_net = self.vgg19.inference(pred_image)
			style_loss = 0.0
			for layer in _STYLE_LAYERS:
				tensor_shape = pred_net[layer].get_shape().as_list()
				pred_reshape = tf.reshape(pred_net[layer],[-1, tensor_shape[-1]])
				dim = 1.0
				for temp_dim in tensor_shape:
					dim *= temp_dim
				pred_transpose = tf.matrix_transpose(pred_reshape)
				pred_feature = tf.matmul(pred_transpose, pred_reshape) / dim
				style_loss += tf.reduce_mean(tf.square(pred_feature - style_feature_dic[layer]))

			content_loss = 0.0
			for layer in _CONTENT_LAYER:
				content_loss += tf.reduce_mean(tf.square(pred_net[layer] - content_feature_dic[layer]))

			# tv loss
			x_loss = tf.reduce_mean(tf.square(pred_image[:,1:,:,:] - pred_image[:,0:-1,:,:]))
			y_loss = tf.reduce_mean(tf.square(pred_image[:,:,1:,:] - pred_image[:,:,0:-1,:]))
			tv_loss = 0.5 * (x_loss + y_loss)

			loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tv_loss

			opt = tf.train.AdamOptimizer(_LEARNING_RATE).minimize(loss)

			# run optimization for generating new image
			session.run(tf.global_variables_initializer())

			print 'start to generate image...'
			for step in xrange(max_iteration):
				_, generated_image, loss_val, style_loss_val, content_loss_val, tv_loss_val = session.run([opt, pred_image, loss, style_loss, content_loss, tv_loss])
				step += 1

				if step % _OUTPUT_INFO_FREQUENCE == 0:
					print 'step: %d loss: %f, style_loss: %f, content_loss: %f, tv_loss: %f'%(step, loss_val, style_loss_val, content_loss_val, tv_loss)

				if step % _OUTPUT_RESULT_FREQUENCE == 0:
					image = np.reshape(generated_image, image_content.shape)
					image = np.uint8(image)
					scipy.misc.imsave('generated_image_%d.png'%(step), image)

			image = np.reshape(generated_image, image_content.shape)
			image = np.uint8(image)
			return image



