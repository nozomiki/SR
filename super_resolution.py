# coding=utf8

import os
import random
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

import super_resolution_utilty as util


class DataSet:
    def __init__(self, cache_dir, filenames, channels=1, scale=1, alignment=0, jpeg_mode=False, max_value=255.0):

        self.count = len(filenames)        #number of images
#		self.group = len(filenames) // 5
        self.image = self.count * [None]

        for i in range(self.count):
            image = util.load_input_image_with_cache(cache_dir, filenames[i], channels=channels,
			                                         scale=scale, alignment=alignment, jpeg_mode=jpeg_mode)
            self.image[i] = image

    def convert_to_batch_images(self, window_size, stride, max_value=255.0):

        batch_images = self.count * [None]           #number of images
        real_images = self.count // 5
        batch_images_count = 0

        for i in range(self.count):
            image = self.image[i]
            if max_value != 255.0:
                image = np.multiply(self.image[i], max_value / 255.0)
            batch_images[i] = util.get_split_images(image, window_size, stride=stride)        #4-D tensor
            batch_images_count += batch_images[i].shape[0]           #number of batch images

        batch_group_num = batch_images_count // 5
        images = [[] for group in range(batch_group_num)]

        group = 0
        for n in range(real_images):
            for j in range(batch_images[n*5].shape[0]):
                for i in range(n*5, n*5+5):
                    images[group].append(batch_images[i][j])        #images = [group][0~4]
                group += 1
                if (group == batch_group_num):
                    break
        images = np.array(images)

        self.image = images
        self.count = batch_group_num

        print("%d group mini-batch images are built." % len(self.image))

    def convert_to_label_batch(self, window_size, stride, max_value=255.0):

        batch_images = self.count * [None]           #number of images
        batch_images_count = 0

        for i in range(self.count):
            image = self.image[i]
            if max_value != 255.0:
                image = np.multiply(self.image[i], max_value / 255.0)
            batch_images[i] = util.get_split_images(image, window_size, stride=stride)        #4-D tensor
            batch_images_count += batch_images[i].shape[0]           #number of batch images

        images = batch_images_count * [None]
        no = 0
        for i in range(self.count):
            for j in range(batch_images[i].shape[0]):
                images[no] = batch_images[i][j]
                no += 1

        self.image = images
        self.count = batch_images_count

        print("%d mini-batch images are built." % len(self.image))


class DataSets:
    def __init__(self, cache_dir, filenames, label_filenames, scale, batch_size, stride_size, width=0, height=0, channels=1,
	             jpeg_mode=False, max_value=255.0):
        self.input = DataSet(cache_dir, filenames, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode)
        self.input.convert_to_batch_images(batch_size/scale, stride_size/scale, max_value=max_value)

        self.true = DataSet(cache_dir, label_filenames, channels=channels, alignment=scale, jpeg_mode=jpeg_mode)
        self.true.convert_to_label_batch(batch_size, stride_size, max_value=max_value)


class SuperResolution:
    def __init__(self, flags, model_name="model"):

        # Model Parameters
        self.lr = flags.initial_lr
        self.lr_decay = flags.lr_decay
        self.lr_decay_epoch = flags.lr_decay_epoch
        self.beta1 = flags.beta1
        self.beta2 = flags.beta2
        self.momentum = flags.momentum
        self.feature_num = flags.feature_num
        self.cnn_size = flags.cnn_size
        self.depth = flags.depth
        self.cnn_stride = 1
        self.lap_depth = flags.lap_depth
        self.inference_depth = flags.inference_depth
        self.batch_num = flags.batch_num
        self.batch_size = flags.batch_size
        self.stride_size = flags.stride_size
        self.optimizer = flags.optimizer
        self.loss_alpha = flags.loss_alpha
        self.loss_alpha_decay = flags.loss_alpha / flags.loss_alpha_zero_epoch
        self.loss_beta = flags.loss_beta
        self.weight_dev = flags.weight_dev
        self.initializer = flags.initializer

        # Image Processing Parameters
        self.scale = flags.scale
        self.resblock_depth = flags.resblock_depth
        self.max_value = flags.max_value
        self.channels = flags.channels
        self.jpeg_mode = flags.jpeg_mode
        self.residual = flags.residual

        # Training or Other Parameters
        self.checkpoint_dir = flags.checkpoint_dir
        self.model_name = model_name

        # Debugging or Logging Parameters
        self.log_dir = flags.log_dir
        self.debug = flags.debug
        self.visualize = flags.visualize
        self.summary = flags.summary
        self.log_weight_image_num = 16

        # initializing variables
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.H_conv = (self.inference_depth + 1) * [None]
        self.HS_conv = self.inference_depth * [None]
        self.batch_input_images = self.batch_num * [None]
        self.batch_true_images = self.batch_num * [None]

        self.index_in_epoch = -1
        self.epochs_completed = 0
        self.min_validation_mse = -1
        self.min_validation_epoch = -1
        self.step = 0
        self.training_psnr = 0

        self.psnr_graph_epoch = []
        self.psnr_graph_value = []

        util.make_dir(self.log_dir)
        util.make_dir(self.checkpoint_dir)
        if flags.initialise_log:
            util.clean_dir(self.log_dir)

        print("Features:%d Inference Depth:%d Initial LR:%0.5f [%s]" % \
              (self.feature_num, self.inference_depth, self.lr, self.model_name))

    def load_datasets(self, cache_dir, training_filenames, train_label_filenames, test_filenames, test_label_filenames, batch_size, stride_size):
        self.train = DataSets(cache_dir, training_filenames, train_label_filenames, self.scale, batch_size, stride_size,
		                      channels=self.channels, jpeg_mode=self.jpeg_mode, max_value=self.max_value)
        self.test = DataSets(cache_dir, test_filenames, test_label_filenames, self.scale, batch_size, stride_size,
		                     channels=self.channels, jpeg_mode=self.jpeg_mode, max_value=self.max_value)

    def set_next_epoch(self):

        self.loss_alpha = max(0, self.loss_alpha - self.loss_alpha_decay)

        self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
        self.epochs_completed += 1
        self.index_in_epoch = 0

    def build_training_batch(self):

        if self.index_in_epoch < 0:
            self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
            self.index_in_epoch = 0

        for i in range(self.batch_num):
            if self.index_in_epoch >= self.train.input.count:
                self.set_next_epoch()

            self.batch_input_images[i] = self.train.input.image[self.batch_index[self.index_in_epoch]]
            self.batch_true_images[i] = self.train.true.image[self.batch_index[self.index_in_epoch]]
            self.index_in_epoch += 1

    def LapSRNSingleLevel(self, net_image, net_feature, shapes, reuse=False):
        with tf.variable_scope("Model_level", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_tmp = net_feature
#			shapes = net_image.outputs.get_shape()
            # recursive block
            for d in range(self.resblock_depth):
                 net_tmp = Conv3dLayer(net_tmp,shape=[1, self.cnn_size, self.cnn_size, self.feature_num, self.feature_num],
                               strides=[1,1,self.cnn_stride, self.cnn_stride,1], name='conv_D%d' % d, act=tf.nn.relu,
                               W_init=tf.contrib.layers.xavier_initializer())

            net_feature = DeConv3dLayer(net_tmp,shape=[1, self.cnn_size, self.cnn_size, self.feature_num, self.feature_num],
                               output_shape=[shapes[0], 5, 2*shapes[2], 2*shapes[3], self.feature_num], act=tf.nn.relu,
                               strides=[1,1,self.cnn_stride*2, self.cnn_stride*2,1], name='upconv_feature',
                               W_init=tf.contrib.layers.xavier_initializer())

            # add image back
            residual_level = Conv3dLayer(net_feature,shape=[1, self.cnn_size, self.cnn_size, self.feature_num, self.channels],
                               strides=[1,1,self.cnn_stride, self.cnn_stride,1], name='res',
                               W_init=tf.contrib.layers.xavier_initializer())

            net_image = UpSampling2dLayer(net_image, size=[2,2], name='upconv_image')
            res_image = DrawOutLayer(residual_level, name='residual_image')
            net_image = ElementwiseLayer(layer=[res_image,net_image],combine_fn=tf.add,name='add_image')

        return net_image, net_feature, residual_level



    def build_lap_graph(self, reuse=False):

        self.x = tf.placeholder(tf.float32, shape=[None, 5, None, None, self.channels], name="X")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="Y")

        n_level = int(np.log2(self.scale))
        assert n_level >= 1

        shapes = tf.shape(self.x)
        with tf.variable_scope("LapSRN", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            inputs_level = InputLayer(self.x, name='input_level')

            net_feature = Conv3dLayer(inputs_level, shape=[1, self.cnn_size, self.cnn_size, self.channels, self.feature_num],
                           strides=[1,1,self.cnn_stride, self.cnn_stride,1], W_init=tf.contrib.layers.xavier_initializer(),
	                        act=tf.nn.relu, name='init_conv')

            image = self.x[:,4,:,:,:]
            net_image = InputLayer(image, name='image_level')

            # 2X for each level
            net_image1, net_feature1, net_residual1 = self.LapSRNSingleLevel(net_image, net_feature, shapes, reuse=reuse)
#			net_image2, net_feature2, net_residual2 = self.LapSRNSingleLevel(net_image1, net_feature1, reuse=True)

        self.net_image = net_image1
        self.net_residual = net_residual1
        self.net_feature = net_feature1
#        if self.visualize:
#            tf.summary.image("residual0/" + self.model_name, self.net_residual.outputs[:,0,:,:,:], max_outputs=1)
#            tf.summary.image("residual1/" + self.model_name, self.net_residual.outputs[:,1,:,:,:], max_outputs=1)
#            tf.summary.image("residual2/" + self.model_name, self.net_residual.outputs[:,2,:,:,:], max_outputs=1)
#            tf.summary.image("residual3/" + self.model_name, self.net_residual.outputs[:,3,:,:,:], max_outputs=1)
#            tf.summary.image("residual4/" + self.model_name, self.net_residual.outputs[:,4,:,:,:], max_outputs=1)

        if self.summary:
            # convert to tf.summary.image format [batch_num, height, width, channels]
            Wm1_transposed = tf.transpose(self.Wm1_conv, [3, 0, 1, 2])
            tf.summary.image("W-1/" + self.model_name, Wm1_transposed, max_outputs=self.log_weight_image_num)
            util.add_summaries("B-1", self.model_name, self.Bm1_conv, mean=True, max=True, min=True)
            util.add_summaries("W-1", self.model_name, self.Wm1_conv, mean=True, max=True, min=True)

            util.add_summaries("B0", self.model_name, self.Bp_conv, mean=True, max=True, min=True)
            util.add_summaries("W0", self.model_name, self.Wp_conv, mean=True, max=True, min=True)

    def build_merge_graph(self):

        # M-2
        self.Wmer2_conv = util.weight([3, 1, 1, self.feature_num, self.feature_num],
		                            stddev=self.weight_dev, name="Wm-2_conv", initializer=self.initializer)
        self.Bmer2_conv = util.bias([self.feature_num], name="Bm-2")
        self.Mer2_conv = util.conv3d_with_bias(self.net_feature.outputs, self.Wmer2_conv, self.cnn_stride, self.Bmer2_conv, pad="VALID", add_relu=True, name="M-2")

        # M-1
        self.Wmer1_conv = util.weight([3, 1, 1, self.feature_num, self.feature_num],
		                            stddev=self.weight_dev, name="Wm-1_conv", initializer=self.initializer)
        self.Bmer1_conv = util.bias([self.feature_num], name="Bm-1")
        self.H_conv[0] = util.conv3d_with_bias(self.Mer2_conv, self.Wmer2_conv, self.cnn_stride, self.Bmer2_conv, pad="VALID", add_relu=True, name="M-1")

        self.H_conv[0] = self.H_conv[0][:,0,:,:,:]

    def build_inference_graph(self):

        if self.inference_depth <= 0:
            return

        self.WL_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.feature_num],
		                          stddev=self.weight_dev, name="WL_conv", initializer="diagonal")
        self.BL_conv = util.bias([self.feature_num], name="BL")

        self.WS_conv = util.weight([self.cnn_size, self.cnn_size, 1, self.feature_num],
		                          stddev=self.weight_dev, name="WS_conv", initializer="diagonal")
        self.BS_conv = util.bias([self.feature_num], name="BS")

        for i in range(0, self.inference_depth):
            self.H_conv[i + 1] = util.conv2d_with_bias(self.H_conv[i], self.WL_conv, 1, self.BL_conv, add_relu=True,
			                                           name="H%d" % (i + 1))
            self.HS_conv[i] = util.conv2d_with_bias(self.net_residual.outputs[:,i,:,:,:], self.WS_conv, 1, self.BS_conv, add_relu=True,
			                                           name="HS%d" % (i + 1))
            self.H_conv[i + 1] = tf.add(self.H_conv[i + 1], self.HS_conv[i])

            # tf.summary.image("Feature_map%d/" % (i+1) + self.model_name, self.R_conv, max_outputs=4)

        self.W_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.channels],
		                          stddev=self.weight_dev, name="W_conv", initializer=self.initializer)
        self.B_conv = util.bias([self.channels], name="B")

        self.H = util.conv2d_with_bias(self.H_conv[self.inference_depth], self.W_conv, 1, self.B_conv, add_relu=True,
			                                           name="H")

        self.y_ = self.H

        tf.summary.image("prediction/" + self.model_name, self.y_, max_outputs=1)

        if self.residual:
            self.y_ = tf.add(self.y_, self.net_image.outputs, name="output")

        if self.summary:
            util.add_summaries("W", self.model_name, self.W_conv, mean=True, max=True, min=True)
            util.add_summaries("B", self.model_name, self.B_conv, mean=True, max=True, min=True)
            util.add_summaries("BD1", self.model_name, self.BD1_conv)
            util.add_summaries("WD1", self.model_name, self.WD1_conv, mean=True, max=True, min=True)
            util.add_summaries("WD2", self.model_name, self.WD2_conv, mean=True, max=True, min=True)

    def build_optimizer(self):

        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.loss_alpha_input = tf.placeholder(tf.float32, shape=[], name="Alpha")

        mse = tf.reduce_mean(tf.square(self.y_ - self.y), name="loss1")
        if self.debug:
            mse = tf.Print(mse, [mse], message="MSE: ")

        tf.summary.scalar("test_PSNR/" + self.model_name, self.get_psnr_tensor(mse))

        if self.loss_alpha == 0.0 or self.inference_depth == 0:
            loss = mse
        else:
            loss1 = tf.reduce_mean(tf.square(self.net_image.outputs - self.y), name="loss1")
            loss2 = mse
            if self.visualize:
                tf.summary.scalar("loss1/" + self.model_name, loss1)
                tf.summary.scalar("loss2/" + self.model_name, loss2)
            loss1 = tf.multiply(self.loss_alpha_input, loss1, name="loss1_alpha")
            loss2 = tf.multiply(1 - self.loss_alpha_input, loss2, name="loss2_alpha")

            loss = loss1 + loss2

        if self.visualize:
            tf.summary.scalar("test_loss/" + self.model_name, loss)

        self.loss = loss
        self.mse = mse
        self.train_step = self.add_optimizer_op(loss, self.lr_input)

        util.print_num_of_total_parameters()

    def get_psnr_tensor(self, mse):

        value = tf.constant(self.max_value, dtype=mse.dtype) / tf.sqrt(mse)
        numerator = tf.log(value)
        denominator = tf.log(tf.constant(10, dtype=mse.dtype))
        return tf.constant(20, dtype=mse.dtype) * numerator / denominator

    def add_optimizer_op(self, loss, lr_input):

        if self.optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr_input).minimize(loss)
        elif self.optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr_input).minimize(loss)
        elif self.optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr_input).minimize(loss)
        elif self.optimizer == "adam":
            train_step = tf.train.AdamOptimizer(lr_input, beta1=self.beta1, beta2=self.beta2).minimize(loss)
        elif self.optimizer == "momentum":
            train_step = tf.train.MomentumOptimizer(lr_input, self.momentum).minimize(loss)
        elif self.optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr_input, momentum=self.momentum).minimize(loss)
        else:
            print("Optimizer arg should be one of [gd, adagrad, adam, momentum, rmsprop].")
            return None

        return train_step

    def init_all_variables(self, load_initial_data=False):

        if self.visualize:
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if load_initial_data:
            self.saver.restore(self.sess, self.checkpoint_dir + "/" + self.model_name + ".ckpt")
            print("Model restored.")

        self.start_time = time.time()

    def train_batch(self, log_mse=False):

        _, mse = self.sess.run([self.train_step, self.mse], feed_dict={self.x: self.batch_input_images,
		                                                               self.y: self.batch_true_images,
		                                                               self.lr_input: self.lr,
		                                                               self.loss_alpha_input: self.loss_alpha})
        self.step += 1
        self.training_psnr = util.get_psnr(mse, max_value=self.max_value)

    def evaluate(self):

        summary_str, mse = self.sess.run([self.summary_op, self.mse],
		                                 feed_dict={self.x: self.test.input.image,
		                                            self.y: self.test.true.image,
		                                            self.loss_alpha_input: self.loss_alpha})
        print("finish")
        self.summary_writer.add_summary(summary_str, self.step)
        self.summary_writer.flush()

        if self.min_validation_mse < 0 or self.min_validation_mse > mse:
            self.min_validation_epoch = self.epochs_completed
            self.min_validation_mse = mse
        else:
            if self.epochs_completed > self.min_validation_epoch + self.lr_decay_epoch:
                self.min_validation_epoch = self.epochs_completed
                self.min_validation_mse = mse
                self.lr *= self.lr_decay

        psnr = util.get_psnr(mse, max_value=self.max_value)
        self.psnr_graph_epoch.append(self.epochs_completed)
        self.psnr_graph_value.append(psnr)

        return mse

    def save_summary(self):

        summary_str = self.sess.run(self.summary_op,
		                            feed_dict={self.x: self.test.input.image,
		                                       self.y: self.test.true.image,
		                                       self.loss_alpha_input: self.loss_alpha})

        self.summary_writer.add_summary(summary_str, 0)
        self.summary_writer.flush()

    def print_status(self, mse):

        psnr = util.get_psnr(mse, max_value=self.max_value)
        if self.step == 0:
            print("Initial MSE:%f PSNR:%f" % (mse, psnr))
        else:
            processing_time = (time.time() - self.start_time) / self.step
            print("%s Step:%d MSE:%f PSNR:%f (%f)" % (util.get_now_date(), self.step, mse, psnr, self.training_psnr))
            print("Epoch:%d LR:%f α:%f (%2.2fsec/step)" % (self.epochs_completed, self.lr, self.loss_alpha, processing_time))

    def print_weight_variables(self):

        util.print_CNN_weight(self.Wm1_conv)
        util.print_CNN_bias(self.Bm1_conv)
        util.print_CNN_weight(self.Wp_conv)
        util.print_CNN_bias(self.Bp_conv)
        util.print_CNN_bias(self.W)

    def save_model(self):

        filename = self.checkpoint_dir + "/" + self.model_name + ".ckpt"
        self.saver.save(self.sess, filename)
        print("Model saved [%s]." % filename)

    def save_all(self):

        self.save_model()

        psnr_graph = np.column_stack((np.array(self.psnr_graph_epoch), np.array(self.psnr_graph_value)))

        filename = self.checkpoint_dir + "/" + self.model_name + ".csv"
        np.savetxt(filename, psnr_graph, delimiter=",")
        print("Graph saved [%s]." % filename)

    def do(self, input_image, label_image):

        if len(input_image.shape) == 3:
            input_image = input_image.reshape(input_image.shape[0], input_image.shape[1], input_image.shape[2], 1)
        if len(label_image.shape) == 2:
            label_image = label_image.reshape(label_image.shape[0], label_image.shape[1], 1)

        image = np.multiply(input_image, self.max_value / 255.0)
        label_image = np.multiply(label_image, self.max_value / 255.0)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], image.shape[3])
        label_image = label_image.reshape(1, label_image.shape[0], label_image.shape[1], label_image.shape[2])
        y = self.sess.run(self.y_, feed_dict={self.x: image})
        print(y.shape)
#        summary_str = self.sess.run(self.summary_op,
#		                            feed_dict={self.x: image,
#		                                       self.y: label_image,
#		                                       self.loss_alpha_input: self.loss_alpha})
#
#        self.summary_writer.add_summary(summary_str, 0)
#        self.summary_writer.flush()

        return np.multiply(y[0], 255.0 / self.max_value)

    def do_super_resolution(self, label_file_path, file_path, output_folder="output"):

        filename, extension = os.path.splitext(label_file_path)
        output_folder = output_folder + "/"
        org_image = util.load_image(label_file_path)
#        util.save_image(output_folder + label_file_path, org_image)

        input_ycbcr_image = []
        if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
            for i in range (0,5):
                input_image = util.load_image(file_path[i])
                scaled_image = util.resize_image_by_pil_bicubic(input_image, 1.0/self.scale)
#                util.save_image(output_folder + file_path[i] + "_bicubic" + extension, scaled_image)
                scaled_image = util.convert_rgb_to_ycbcr(scaled_image, jpeg_mode=self.jpeg_mode)
                input_ycbcr_image.append(scaled_image[:, :, 0:1])
            input_ycbcr_image = np.array(input_ycbcr_image)

            org_image = util.convert_rgb_to_ycbcr(org_image, jpeg_mode=self.jpeg_mode)
            output_y_image = self.do(input_ycbcr_image, org_image[:, :, 0:1])
#            util.save_image(output_folder + filename + "_result_y" + extension, output_y_image)

            img_up = util.resize_image_by_pil_bilinear(input_ycbcr_image[4], self.scale)
            mse_bic = util.compute_mse(org_image[:, :, 0:1], img_up, border_size=self.scale)
            mse = util.compute_mse(org_image[:, :, 0:1], output_y_image, border_size=self.scale)

            image = util.convert_y_and_cbcr_to_rgb(output_y_image, org_image[:, :, 1:3], jpeg_mode=self.jpeg_mode)
        elif len(org_image.shape) >= 3 and org_image.shape[2] == 1:
            for i in range (0,5):
                input_image = util.load_image(file_path[i])
                scaled_image = util.resize_image_by_pil_bicubic(input_image, 1.0/self.scale)
#                util.save_image(output_folder + file_path[i] + "_bicubic" + extension, scaled_image)
                input_ycbcr_image.append(scaled_image)
            input_ycbcr_image = np.array(input_ycbcr_image)

            image = self.do(input_ycbcr_image, org_image)

            img_up = util.resize_image_by_pil_bilinear(input_ycbcr_image[4], self.scale)
            mse_bic = util.compute_mse(org_image, img_up, border_size=self.scale)
            mse = util.compute_mse(org_image, image, border_size=self.scale)
        else:
            print('wrong')
            return 0

#        util.save_image(output_folder + filename + "_result" + extension, image)
        return mse_bic, mse

    def do_super_resolution_for_test(self, i, label_file_path, file_path, output_folder="output", output=True):

        true_image = util.set_image_alignment(util.load_image(label_file_path), self.scale)
        output_folder = output_folder + "/"
        filename, extension = os.path.splitext(label_file_path)

        input_y_image = []
        if len(true_image.shape) >= 3 and true_image.shape[2] == 3 and self.channels == 1:
            for j in range (i*5, i*5+5):
                input_image = util.load_input_image(file_path[i], channels=1, scale=self.scale, alignment=self.scale)
                input_y_image.append(input_image)         # convert_ycbcr:True->False
            input_y_image = np.dstack((input_y_image[0], input_y_image[1],
                                       input_y_image[2], input_y_image[3], input_y_image[4]))
            true_ycbcr_image = util.convert_rgb_to_ycbcr(true_image, jpeg_mode=self.jpeg_mode)

            output_y_image = self.do(input_y_image, true_ycbcr_image[:, :, 0:1])
            mse = util.compute_mse(true_ycbcr_image[:, :, 0:1], output_y_image, border_size=self.scale)

            if output:
                output_color_image = util.convert_y_and_cbcr_to_rgb(output_y_image, true_ycbcr_image[:, :, 1:3],
        		                                                    jpeg_mode=self.jpeg_mode)
                loss_image = util.get_loss_image(true_ycbcr_image[:, :, 0:1], output_y_image, border_size=self.scale)

                util.save_image(output_folder + label_file_path, true_image)
                util.save_image(output_folder + filename + "_input" + extension, input_y_image)
                util.save_image(output_folder + filename + "_true_y" + extension, true_ycbcr_image[:, :, 0:1])
                util.save_image(output_folder + filename + "_result" + extension, output_y_image)
                util.save_image(output_folder + filename + "_result_c" + extension, output_color_image)
                util.save_image(output_folder + filename + "_loss" + extension, loss_image)
        else:
            for j in range (i*5, i*5+5):
                input_image = util.load_input_image(file_path[i], channels=1, scale=self.scale, alignment=self.scale)
                input_y_image.append(util.build_input_image(input_image, channels=self.channels, scale=self.scale, alignment=self.scale,
        	                                                convert_ycbcr=False, jpeg_mode=self.jpeg_mode))         # convert_ycbcr:True->False
            input_y_image = np.dstack((input_y_image[0], input_y_image[1],
                                       input_y_image[2], input_y_image[3], input_y_image[4]))
            output_image = self.do(input_y_image, true_image)
            mse = util.compute_mse(true_image, output_image, border_size=self.scale)

            if output:
                util.save_image(output_folder + label_file_path, true_image)
                util.save_image(output_folder + filename + "_result" + extension, output_image)

        print("MSE:%f PSNR:%f" % (mse, util.get_psnr(mse)))
        return mse

    def end_train_step(self):
        self.total_time = time.time() - self.start_time

    def print_steps_completed(self):
        if self.step <= 0:
            return

        processing_time = self.total_time / self.step

        h = self.total_time // (60 * 60)
        m = (self.total_time - h * 60 * 60) // 60
        s = (self.total_time - h * 60 * 60 - m * 60)

        print("Finished at Total Epoch:%d Step:%d Time:%02d:%02d:%02d (%0.3fsec/step)" % (
            self.epochs_completed, self.step, h, m, s, processing_time))
