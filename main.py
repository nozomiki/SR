# coding=utf8

"""
Deeply-Recursive Convolutional Network for Image Super-Resolution
by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea

Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html

Test implementation using TensorFlow library.

Author: Jin Yamanaka
Many thanks for: Masayuki Tanaka and Shigesumi Kuwashima
Project: https://github.com/jiny2001/deeply-recursive-cnn-tf
"""

import tensorflow as tf
import super_resolution as sr
import super_resolution_utilty as util

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_float("initial_lr", 5e-5, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 1, "Decay learning rate when loss does not decrease")
flags.DEFINE_float("beta1", 0.1, "Beta1 form adam optimizer")
flags.DEFINE_float("beta2", 0.1, "Beta2 form adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("feature_num", 64, "Number of CNN Filters")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("depth", 5, "Input dedpth")
flags.DEFINE_integer("lap_depth", 3, "Number of CNN filters")
flags.DEFINE_integer("inference_depth", 5, "Number of recurrent CNN filters")
flags.DEFINE_integer("batch_num", 65, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_size", 64, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 40, "Stride size for mini-batch")
flags.DEFINE_string("optimizer", "momentum", "Optimizer: can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("loss_alpha", 0.3, "Initial loss-alpha value (0-1). Don't use intermediate outputs when 0.")
flags.DEFINE_integer("loss_alpha_zero_epoch", 6, "Decrease loss-alpha to zero by this epoch")
flags.DEFINE_float("loss_beta", 0.0001, "Loss-beta for weight decay")
flags.DEFINE_float("weight_dev", 0.001, "Initial weight stddev")
flags.DEFINE_string("initializer", "he", "Initializer: can be [uniform, stddev, diagonal, xavier, he]")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale for Super Resolution (can be 2 or 4)")
flags.DEFINE_integer("resblock_depth", 3, "Depth for each level of the LapSRN graph")
flags.DEFINE_float("max_value", 255.0, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Using num of image channels. Use YCbCr when channels=1.")
flags.DEFINE_boolean("jpeg_mode", False, "Using Jpeg mode for converting from rgb to ycbcr")
flags.DEFINE_boolean("residual", True, "Using residual net")

# Training or Others
flags.DEFINE_boolean("is_training", True, "Train model with 91 standard images")
flags.DEFINE_string("dataset", "set5", "Test dataset. [set5, set14, bsd100, urban100, all, test] are available")
flags.DEFINE_string("label_dataset", "set5_label", "Test dataset. [set5, set14, bsd100, urban100, all, test] are available")
flags.DEFINE_string("training_set", "y4m", "Training dataset. [ScSR, Set5, Set14, Bsd100, Urban100,y4m] are available")
flags.DEFINE_string("training_label", "label", "Training label dataset")
flags.DEFINE_integer("evaluate_step", 10, "steps for evaluation")
flags.DEFINE_integer("save_step", 50, "steps for saving learned model")
flags.DEFINE_float("end_lr", 1e-6, "Training end learning rate")
flags.DEFINE_string("checkpoint_dir", "model", "Directory for checkpoints")
flags.DEFINE_string("cache_dir", "cache", "Directory for caching image data. If specified, build image cache")
flags.DEFINE_string("data_dir", "data", "Directory for test/train images")
flags.DEFINE_boolean("load_model", True, "Load saved model before start")
flags.DEFINE_string("model_name", "F64_D5_LR0.000100", "model name for save files and tensorboard log")

# Debugging or Logging
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_boolean("debug", False, "Display each calculated MSE and weight variables")
flags.DEFINE_boolean("initialise_log", False, "Clear all tensorboard log before start")
flags.DEFINE_boolean("visualize", True, "Save loss and graph data")
flags.DEFINE_boolean("summary", False, "Save weight and bias")


def main(_):

  print("Super Resolution (tensorflow version:%s)" % tf.__version__)
  print("%s\n" % util.get_now_date())

  if FLAGS.model_name is "":
    model_name = "model_F%d_D%d_LR%f" % (FLAGS.feature_num, FLAGS.inference_depth, FLAGS.initial_lr)
  else:
    model_name = "model_%s" % FLAGS.model_name
  model = sr.SuperResolution(FLAGS, model_name=model_name)

  test_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale)
  test_label_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.label_dataset, FLAGS.scale)
  if FLAGS.is_training:
    if FLAGS.dataset == "test":
      training_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale)
      train_label_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.label_dataset, FLAGS.scale)
    else:
      training_filenames =  util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.training_set + "/")
      train_label_filenames =  util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.training_label + "/")      

    print("Loading and building cache images...")
    model.load_datasets(FLAGS.cache_dir, training_filenames, train_label_filenames, test_filenames, test_label_filenames, FLAGS.batch_size, FLAGS.stride_size)
  else:
    FLAGS.load_model = True

  model.build_lap_graph()
  model.build_merge_graph()
  model.build_inference_graph()
  model.build_optimizer()
  model.init_all_variables(load_initial_data=FLAGS.load_model)

  if FLAGS.is_training:
    train(training_filenames, test_filenames, model)
  
  psnr = 0
  total_mse = 0
  i = 0
  for filename in test_label_filenames:
    mse = model.do_super_resolution_for_test(i, filename, test_filenames, FLAGS.output_dir)
    total_mse += mse
    psnr += util.get_psnr(mse)
    i += 1

  print ("\n--- summary --- %s" % util.get_now_date())
  model.print_steps_completed()
  util.print_num_of_total_parameters()
  print("Final MSE:%f, PSNR:%f" % (total_mse / len(test_filenames), psnr / len(test_filenames)))

  
def train(training_filenames, test_filenames, model):

  mse = model.evaluate()
  model.print_status(mse)

  while model.lr > FLAGS.end_lr:
  
    logging = model.step % FLAGS.evaluate_step == 0     # %表示取余
    model.build_training_batch()
    model.train_batch(log_mse=logging)

    if logging:
      mse = model.evaluate()
      model.print_status(mse)

    if model.step > 0 and model.step % FLAGS.save_step == 0:
      model.save_model()

  model.end_train_step()
  model.save_all()

  if FLAGS.debug:
    model.print_weight_variables()
    

if __name__ == '__main__':
  tf.app.run()
