import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import model_80 as model
import batchdata_generator_80 as DG

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 2]')
parser.add_argument('--model_dir', default='train_log_80_5000veh', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--testing_data_path', default='./data_80_80_2000_implementing/', help='Make sure the source training-data files path')
FLAGS = parser.parse_args()

TEST_DATA_PATH = FLAGS.testing_data_path
MODEL_DIR = FLAGS.model_dir
BATCH_SIZE = FLAGS.batch_size

NUM_R = 80
NUM_C = 80

def test():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            with tf.device('/gpu:0'):
                input = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 12))
                label = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                mask = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                is_training = tf.compat.v1.placeholder(tf.bool, shape=())
                pre = model.Unet('unet', input, training=is_training)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),  max_to_keep=3)
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=config)
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(MODEL_DIR)) # load model
        train_set, val_set = DG.get_train_val_set(TEST_DATA_PATH)
        train_generator = DG.generate_training_minibatch_data(TEST_DATA_PATH, BATCH_SIZE, train_set)

        ops = {'input': input,
               'is_training': is_training,
               'label': label,
               'mask': mask,
               'pre': pre}
        test_data_set = os.listdir(TEST_DATA_PATH)
        test_one_sample(sess, ops, train_generator, test_data_set)

def test_one_sample(sess, ops, generator, test_data_set):
    """ ops: dict mapping from string to tf ops """

    print('-----------------testing--------------------')
    for i in tqdm(range(len(test_data_set))):
        ###

        temp_data_dir = TEST_DATA_PATH + test_data_set[i]
        try:
            batch_test_data = np.load(temp_data_dir)
        except:
            batch_test_data = np.loadtxt(temp_data_dir)

        batch_test_data, label = next(generator)
        # batch_test_data = np.expand_dims(batch_test_data, 0)
        ###
        batch_mask_data = batch_test_data[:, :, :, 10].reshape(1, 80, 80, 1)
        batch_mask_data = np.where(batch_mask_data > 0, 1, batch_mask_data)
        feed_dict = {ops['input']: batch_test_data,
                     ops['label']: label,
                     ops['mask']: batch_mask_data,
                     ops['is_training']: False}

        # print('-----label-----',label.sum())
        # print('___label___sum____', label.sum())
        # print('___error___sum___',(label-pre).sum())
        # print(label.sum())
        # print((pre-label).sum())
        pre_, mask_, input_ = sess.run([ops['pre'], ops['mask'], ops['input']], feed_dict=feed_dict)
        input_ = np.squeeze(input_)
        input_ = np.reshape(input_,(6400,12))
        print(input_)
        pre_ = pre_*batch_mask_data
        pre__ = pre_.flatten()
        pre__ = np.reshape(pre__, (-1,1))
        lab__ = np.reshape(label, (-1,1))
        result = np.hstack((input_,lab__))
        result = np.hstack((result, pre__))
        my_string = temp_data_dir
        split_strings = my_string.split('/')
        split_strings.pop(1)
        split_strings.insert(1, 'result')
        final_string = '/'.join(split_strings)
        np.save(final_string, result)
        pre_ = np.squeeze(pre_)
        lab_ = np.squeeze(label)
        print('error^', abs(pre_ - lab_).sum()/3420)
        # print(abs(pre_.sum()-label.sum())/13108)
        # print('loss:', (np.square((lab_ - pre_)*mask_)).sum()/13108.0)
        # print('-----pre-----',pre_.sum())


if __name__ == "__main__":
    test()


