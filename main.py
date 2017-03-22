import numpy as np
import tensorflow as tf
import scipy.misc
from PIL import Image
from style_transfer import StyleTransfer

_CONTENT_IMAGE_PATH = './image/chicago.jpg'
_STYLE_IMAGE_PATH = './image/la_muse.jpg'
_PRETRAINED_VGG19_MODEL = './pretrained_model/vgg19.npy'
_SAVE_IMAGE_PATH = './image/generated_image_final.png'

_IMAGE_SIZE = 300

def main():
	# read image from file
	content_image = scipy.misc.imread(_CONTENT_IMAGE_PATH).astype(np.float)
	content_image = scipy.misc.imresize(content_image, [_IMAGE_SIZE, _IMAGE_SIZE])
	style_image = scipy.misc.imread(_STYLE_IMAGE_PATH).astype(np.float)
	style_image = scipy.misc.imresize(style_image, content_image.shape[:2])

	init_image = scipy.misc.imread(_SAVE_IMAGE_PATH).astype(np.float)
	init_image = scipy.misc.imresize(init_image, content_image.shape[:2])
	# style transfer model
	image_style_transfer = StyleTransfer(_PRETRAINED_VGG19_MODEL)

	generated_image = image_style_transfer.image_style_transfer(content_image, style_image, 5, 0.2, 10, init_image = init_image)
	scipy.misc.imsave(_SAVE_IMAGE_PATH, generated_image)

if __name__ == '__main__':
	main()