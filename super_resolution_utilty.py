# coding=utf8

from __future__ import division

import datetime
import math
import os
import shutil
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc

# utilities for save / load

test_datasets = {
	"set5": ["Set5", 0, 25],
    "set5_label": ["Set5_label", 0, 5],
	"set14": ["Set14", 0, 70],
    "set14_label": ["Set14_label", 0, 14],
	"bsd100": ["BSD100", 0, 500],
    "bsd100_label": ["BSD100_label", 0, 100],
	"urban100": ["Urban100", 0, 500],
    "urban100_label": ["Urban100_label", 0, 100],
	"test": ["Set5", 0, 1],
	"test1.1": ["Test1.1", 0, 25],
	 "test1.1_label": ["Test1.1_label", 0, 5]
}


class LoadError(Exception):
	def __init__(self, message):
		self.message = message


def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


def get_files_in_directory(path):
	file_list = [path + f for f in sorted(listdir(path)) if isfile(join(path, f))]
	return file_list


def remove_generic(path, __func__):
	try:
		__func__(path)
	except OSError as error:
		print("OS error: {0}".format(error))


def clean_dir(path):
	if not os.path.isdir(path):
		return

	files = os.listdir(path)
	for x in files:
		full_path = os.path.join(path, x)
		if os.path.isfile(full_path):
			f = os.remove
			remove_generic(full_path, f)
		elif os.path.isdir(full_path):
			clean_dir(full_path)
			f = os.rmdir
			remove_generic(full_path, f)


def save_image(filename, image):
	if len(image.shape) >= 3 and image.shape[2] == 1:
		image = image.reshape(image.shape[0], image.shape[1])

	directory = os.path.dirname(filename)
	if directory != "" and not os.path.exists(directory):
		os.makedirs(directory)

	image = misc.toimage(image, cmin=0, cmax=255)  # to avoid range rescaling
	misc.imsave(filename, image)

	print("Saved [%s]" % filename)


def save_image_data(filename, image):
	directory = os.path.dirname(filename)
	if directory != "" and not os.path.exists(directory):
		os.makedirs(directory)

	np.save(filename, image)
	print("Saved [%s]" % filename)

	if len(image.shape) == 3 and image.shape[2] == 1:
		image = image.reshape(image.shape[0], image.shape[1])
	misc.imsave(filename, image)


def convert_rgb_to_y(image, jpeg_mode=True, max_value=255.0):
	if len(image.shape) <= 2 or image.shape[2] == 1:
		return image

	if jpeg_mode:
		xform = np.array([[0.299, 0.587, 0.114]])
		y_image = image.dot(xform.T)
	else:
		xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
		y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

	return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=True, max_value=255):
	if len(image.shape) < 2 or image.shape[2] == 1:
		return image

	if jpeg_mode:
		xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
		ycbcr_image = image.dot(xform.T)
		ycbcr_image[:, :, [1, 2]] += max_value / 2
	else:
		xform = np.array(
			[[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
			 [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
		ycbcr_image = image.dot(xform.T)
		ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
		ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

	return ycbcr_image


#def convert_y_and_cbcr_to_ycbcr(y_image, cbcr_image, jpeg_mode=True, max_value=255.0):
#	if len(y_image.shape) <= 2:
#		y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]
#
#	if len(y_image.shape) == 3 and y_image.shape[2] == 3:
#		y_image = y_image[:, :, 0:1]
#
#	ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
#	ycbcr_image[:, :, 0] = y_image[:, :, 0]
#	ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]
#
#	return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=True, max_value=255.0):
	if len(y_image.shape) <= 2:
		y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

	if len(y_image.shape) == 3 and y_image.shape[2] == 3:
		y_image = y_image[:, :, 0:1]

	ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
	ycbcr_image[:, :, 0] = y_image[:, :, 0]
	ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

	return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=True, max_value=255.0):
	rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

	if jpeg_mode:
		rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
		xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
		rgb_image = rgb_image.dot(xform.T)
	else:
		rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
		rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
		xform = np.array(
			[[max_value / 219.0, 0, max_value * 0.701 / 112.0],
			 [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587), - max_value * 0.701 * 0.299 / (112 * 0.587)],
			 [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
		rgb_image = rgb_image.dot(xform.T)

	return rgb_image


def set_image_alignment(image, alignment):
	alignment = int(alignment)  # I don't like this...
	width, height = image.shape[1], image.shape[0]
	width = (width // alignment) * alignment
	height = (height // alignment) * alignment
	if image.shape[1] != width or image.shape[0] != height:
		return image[:height, :width, :]

	return image


def resize_image_by_pil_bilinear(image, scale):
	width, height = image.shape[1], image.shape[0]
	new_width = int(width * scale)
	new_height = int(height * scale)

	if len(image.shape) == 3 and image.shape[2] == 3:
		image = Image.fromarray(image)
		image = image.resize([new_width, new_height], resample=Image.BILINEAR)
		image = np.asarray(image)
	else:
		image = Image.fromarray(image.reshape(height, width))
		image = image.resize([new_width, new_height], resample=Image.BILINEAR)
		image = np.asarray(image)
		image = image.reshape(new_height, new_width, 1)
	return image


def resize_image_by_pil_bicubic(image, scale):
	width, height = image.shape[1], image.shape[0]
	new_width = int(width * scale)
	new_height = int(height * scale)

	if len(image.shape) == 3 and image.shape[2] == 3:
		image = Image.fromarray(image)
		image = image.resize([new_width, new_height], resample=Image.BICUBIC)
		image = np.asarray(image)
	else:
		image = Image.fromarray(image.reshape(height, width))
		image = image.resize([new_width, new_height], resample=Image.BICUBIC)
		image = np.asarray(image)
		image = image.reshape(new_height, new_width, 1)
	return image


def load_image(filename, width=0, height=0, channels=0, alignment=0):
	if not os.path.isfile(filename):
		raise LoadError("File not found [%s]" % filename)
	image = misc.imread(filename)                         #Read an image from a file as an array.

	if len(image.shape) == 2:
		image = image.reshape(image.shape[0], image.shape[1], 1)
	if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
		raise LoadError("Attributes mismatch")
	if channels != 0 and image.shape[2] != channels:
		raise LoadError("Attributes mismatch")
	if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
		raise LoadError("Attributes mismatch")

	print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
	return image


def load_image_data(filename, width=0, height=0, channels=0, alignment=0):
	if not os.path.isfile(filename + ".npy"):
		raise LoadError("File not found")
	image = np.load(filename + ".npy")

	if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
		raise LoadError("Attributes mismatch")
	if channels != 0 and image.shape[2] != channels:
		raise LoadError("Attributes mismatch")
	if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
		raise LoadError("Attributes mismatch")

	print("Cache Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
	return image


def load_input_image(filename, width=0, height=0, channels=1, scale=1, alignment=0,
                     convert_ycbcr=True, jpeg_mode=False, rescale=False):
	image = load_image(filename)
	return build_input_image(image, width, height, channels, scale, alignment,
	                         convert_ycbcr, jpeg_mode, rescale)


def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0,
                      convert_ycbcr=True, jpeg_mode=False, rescale=False):
	if width != 0 and height != 0:
		if image.shape[0] != height or image.shape[1] != width:
			x = (image.shape[1] - width) // 2
			y = (image.shape[0] - height) // 2
			image = image[y: y + height, x: x + width, :]

	if alignment > 1:
		image = set_image_alignment(image, alignment)

   #预处理部分
	if scale != 1:
		image = resize_image_by_pil_bicubic(image, 1.0 / scale)         #下采样
		if rescale:                                                     #
			image = resize_image_by_pil_bicubic(image, scale)           #插值

	if convert_ycbcr:
		image = convert_rgb_to_ycbcr(image, jpeg_mode=jpeg_mode)

	if channels == 1 and image.shape[2] > 1:
		image = image[:, :, 0:1].copy()  # use copy() since after the step we use stride_tricks.as_strided().

	return image


def load_input_image_with_cache(cache_dir, org_filename, channels=1, scale=1, alignment=0,
                                convert_ycbcr=True, jpeg_mode=False, rescale=False):    #convert_ycbcr,rescale:True->False
	if cache_dir is None or cache_dir is "":
		return load_input_image(org_filename, channels=channels, scale=scale, alignment=alignment,
		                        convert_ycbcr=convert_ycbcr, jpeg_mode=jpeg_mode, rescale=rescale)

	filename, extension = os.path.splitext(org_filename)

	if filename.startswith("../"):
		filename = filename[len("../"):]       #[num:]表示num以后的全部元素

	if scale != 1.0:
		filename += "_%1.0f" % scale
	if channels == 1:
		filename += "_Y"

	cache_filename = cache_dir + "/" + filename + extension    #extension=后缀名
	try:
		image = load_image(cache_filename, channels=channels)
	except LoadError:
		image = load_input_image(org_filename, channels=channels, scale=scale, alignment=alignment,
		                         convert_ycbcr=convert_ycbcr, jpeg_mode=jpeg_mode, rescale=rescale)
		save_image(cache_filename, image)

	return image


def get_split_images(image, window_size, stride=None):
	if len(image.shape) == 3 and image.shape[2] == 1:
		image = image.reshape(image.shape[0], image.shape[1])

	window_size = int(window_size)
	size = image.itemsize  # byte size of each value
	height, width = image.shape
	if stride is None:
		stride = window_size
	else:
		stride = int(stride)

	new_height = 1 + (height - window_size) // stride  #返回商的整数
	new_width = 1 + (width - window_size) // stride

	shape = (new_height, new_width, window_size, window_size)
	strides = size * np.array([width * stride, stride, width, 1])
	windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)                #分割 batch
   #reshape  [shape = (batch_num, batch_size, batch_size, channels)]
	windows = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3], 1)

	return windows


# utilities for building graphs

def conv2d(x, w, stride, name=""):
	return tf.nn.conv2d(x, w, strides=[stride, stride, 1, 1], padding="SAME", name=name + "_conv")

def conv3d(x, w, stride, pad="SAME", name=""):
	return tf.nn.conv3d(x, w, strides=[1, 1, stride, stride, 1], padding=pad, name=name + "_conv")

def conv3d_transpose(x, w, stride, name=""):
#	outputs_shape = x.get_shape().as_list()
#	if outputs_shape[0] == None:
#		outputs_shape[0] = 127
#	outputs_shape = [outputs_shape[0], 64, 64, int(w.shape[2])]
	outputs_shape = tf.shape(x)
	outputs_shape = [outputs_shape[0], outputs_shape[1], 2*outputs_shape[2], 2*outputs_shape[3], int(w.shape[3])]
	return tf.nn.conv3d_transpose(x, w, strides=[1, 1, stride, stride, 1], output_shape=outputs_shape,
                               padding="SAME", name=name + "_transp")

def conv2d_with_bias(x, w, stride, bias, name=""):
	conv = conv2d(x, w, stride, name)
	return tf.add(conv, bias, name=name + "_add")


def conv2d_with_bias(x, w, stride, bias, add_relu=False, name=""):
	conv = conv2d(x, w, stride, name)
	if add_relu:
		return tf.nn.relu(tf.add(conv, bias, name=name + "_add"), name=name + "_relu")
	else:
		return tf.add(conv, bias, name=name + "_add")

def conv3d_with_bias(x, w, stride, bias, pad="SAME", add_relu=False, name=""):
	conv = conv3d(x, w, stride, pad, name)
	if add_relu:
		return tf.nn.relu(tf.add(conv, bias, name=name + "_add"), name=name + "_relu")
	else:
		return tf.add(conv, bias, name=name + "_add")


def conv3d_transpose_with_bias(x, w, stride, bias, add_relu=False, name=""):
	conv = conv3d_transpose(x, w, stride, name)
	if add_relu:
		return tf.nn.relu(tf.add(conv, bias, name=name + "_add"), name=name + "_relu")
	else:
		return tf.add(conv, bias, name=name + "_add")


def dilated_conv2d_with_bias(x, w, stride, bias, add_relu=False, name=""):
	conv = tf.nn.atrous_conv2d(x, w, 2, padding="SAME", name=name + "_conv")
	if add_relu:
		return tf.nn.relu(tf.add(conv, bias, name=name + "_add"), name=name + "_relu")
	else:
		return tf.add(conv, bias, name=name + "_add")


def xavier_cnn_initializer(shape, uniform=True, name=None):
	fan_in = shape[0] * shape[1] * shape[2]
	fan_out = shape[0] * shape[1] * shape[3]
	n = fan_in + fan_out
	if uniform:
		init_range = math.sqrt(6.0 / n)
		return tf.random_uniform(shape, minval=-init_range, maxval=init_range, name=name)
	else:
		stddev = math.sqrt(3.0 / n)
		return tf.truncated_normal(shape=shape, stddev=stddev, name=name)


def he_initializer(shape, name=None):
	n = shape[0] * shape[1] * shape[2]
	stddev = math.sqrt(2.0 / n)
	return tf.truncated_normal(shape=shape, stddev=stddev, name=name)


def weight(shape, stddev=0.01, name=None, uniform=False, initializer="xavier"):
	if initializer == "xavier":
		initial = xavier_cnn_initializer(shape, uniform=uniform, name=name)
	elif initializer == "he":
		initial = he_initializer(shape, name=name)
	elif initializer == "uniform":
		initial = tf.random_uniform(shape, minval=-2.0 * stddev, maxval=2.0 * stddev)
	elif initializer == "stddev":
		initial = tf.truncated_normal(shape=shape, stddev=stddev)
	elif initializer == "diagonal":
		initial = tf.truncated_normal(shape=shape, stddev=stddev)
		if len(shape) == 4:
			initial = initial.eval()
			i = shape[0] // 2
			j = shape[1] // 2
			for k in range(min(shape[2], shape[3])):
				initial[i][j][k][k] = 1.0
	else:
		initial = tf.zeros(shape)

	return tf.Variable(initial, name=name)


def bias(shape, initial_value=0.0, name=None):
	if name is None:
		initial = tf.constant(initial_value, shape=shape)
	else:
		initial = tf.constant(initial_value, shape=shape, name=name)
	return tf.Variable(initial)


# utilities for logging -----

def add_summaries(scope_name, model_name, var, stddev=True, mean=False, max=False, min=False):
	with tf.name_scope(scope_name):

		mean_var = tf.reduce_mean(var)
		if mean:
			tf.summary.scalar("mean/" + model_name, mean_var)

		if stddev:
			stddev_var = tf.sqrt(tf.reduce_sum(tf.square(var - mean_var)))
			tf.summary.scalar("stddev/" + model_name, stddev_var)

		if max:
			tf.summary.scalar("max/" + model_name, tf.reduce_max(var))

		if min:
			tf.summary.scalar("min/" + model_name, tf.reduce_min(var))
		tf.summary.histogram(model_name, var)


def get_now_date():
	d = datetime.datetime.today()
	return "%s/%s/%s %s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_loss_image(image1, image2, scale=1.0, border_size=0):
	if len(image1.shape) == 2:
		image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
	if len(image2.shape) == 2:
		image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

	if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
		return None

	if image1.dtype == np.uint8:
		image1 = image1.astype(np.double)
	if image2.dtype == np.uint8:
		image2 = image2.astype(np.double)

	loss_image = np.multiply(np.square(np.subtract(image1, image2)), scale)
	loss_image = np.minimum(loss_image, 255.0)
	loss_image = loss_image[border_size:-border_size, border_size:-border_size, :]

	return loss_image


def compute_mse(image1, image2, border_size=0):
	if len(image1.shape) == 2:
		image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
	if len(image2.shape) == 2:
		image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

	if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
		return None

	if image1.dtype == np.uint8:
		image1 = image1.astype(np.double)
	if image2.dtype == np.uint8:
		image2 = image2.astype(np.double)

	mse = 0.0
	for i in range(border_size, image1.shape[0] - border_size):
		for j in range(border_size, image1.shape[1] - border_size):
			for k in range(image1.shape[2]):
				error = image1[i, j, k] - image2[i, j, k]
				mse += error * error

	return mse / ((image1.shape[0] - 2 * border_size) * (image1.shape[1] - 2 * border_size) * image1.shape[2])


def print_CNN_weight(tensor):
	print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
	weight = tensor.eval()
	for i in range(weight.shape[3]):
		values = ""
		for x in range(weight.shape[0]):
			for y in range(weight.shape[1]):
				for c in range(weight.shape[2]):
					values += "%2.3f " % weight[y][x][c][i]
		print(values)
	print("\n")


def print_CNN_bias(tensor):
	print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
	bias = tensor.eval()
	values = ""
	for i in range(bias.shape[0]):
		values += "%2.3f " % bias[i]
	print(values + "\n")


def get_test_filenames(data_folder, dataset, scale):
	test_folder = data_folder + "/" + test_datasets[dataset][0] +"/"

	test_filenames = []
	for i in range(test_datasets[dataset][1], test_datasets[dataset][2]):
		test_filenames.append(test_folder + "img_%03d.png" % (i + 1))

	return test_filenames


def build_test_filenames(data_folder, dataset, scale):
	test_filenames = []

	if dataset == "all":
		for test_dataset in test_datasets:
			test_filenames += get_test_filenames(data_folder, test_dataset, scale)
	else:
		test_filenames += get_test_filenames(data_folder, dataset, scale)

	return test_filenames


def get_psnr(mse, max_value=255.0):
	if mse is None or mse == float('Inf') or mse == 0:
		psnr = 0
	else:
		psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
	return psnr


def print_num_of_total_parameters():
	total_parameters = 0
	parameters_string = ""
	for variable in tf.trainable_variables():

		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		parameters_string += ("%s-%d, " % (str(shape), variable_parameters))

	print(parameters_string)
	print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


# utility for extracting target files from datasets
def main():
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	flags.DEFINE_string("org_data_folder", "org_data", "Folder for original datasets")
	flags.DEFINE_string("test_set", "all", "Test dataset. set5, set14, bsd100, urban100 or all are available")
	flags.DEFINE_integer("scale", 4, "Scale for Super Resolution (can be 2 or 4)")

	test_filenames = build_test_filenames(FLAGS.org_data_folder, FLAGS.test_set, FLAGS.scale)

	for filename in test_filenames:
		target_filename = "data/" + filename
		print("[%s] > [%s]" % (filename, target_filename))
		if not os.path.exists(os.path.dirname(target_filename)):
			os.makedirs(os.path.dirname(target_filename))
		shutil.copy(filename, target_filename)

	print("OK.")


if __name__ == '__main__':
	main()
