import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm
import batchdata_generator_80 as DG
import model_80 as model
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
tf.random.set_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 2]')
parser.add_argument('--log_dir', default='train_log_80_2000veh', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=48, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=1000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--training_data_path', default='./data_80_80_2000_training/', help='Make sure the source training-data files path')
FLAGS = parser.parse_args()

TRAIN_DATA_PATH = FLAGS.training_data_path
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

NUM_R = 80
NUM_C = 80

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        batch = tf.Variable(0, trainable=False)
        learning_rate = get_learning_rate(batch)

        ####
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            with tf.device('/gpu:0'):

                input = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 12))
                label = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                mask = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                is_training = tf.compat.v1.placeholder(tf.bool, shape=())
                pre = model.Unet('unet', input, training=is_training)
                loss = model.get_loss(pre, label, mask)
                ###optimizers
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),  max_to_keep=3)

        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=config) # config=config
        # Init variables for two GPUs
        init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init)

        ops = {'learning_rate': learning_rate,
               'pre': pre,
               'input': input,
               'label': label,
               'mask':mask,
               'loss': loss,
               'is_training': is_training,
               'train_op': train_op,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_set, val_set = DG.get_train_val_set(TRAIN_DATA_PATH, val_rate=0.10)
            train_generator = DG.generate_training_minibatch_data(TRAIN_DATA_PATH, BATCH_SIZE, train_set)
            val_generator = DG.generate_training_minibatch_data(TRAIN_DATA_PATH, BATCH_SIZE, val_set)

            train_one_epoch(sess, epoch, train_set, train_generator, ops)
            val_one_epoch(sess, epoch, val_set, val_generator, ops)
            # Save the variables to disk.
            saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))

def train_one_epoch(sess, epoch, train_set, generator, ops):
    """ ops: dict mapping from string to tf ops """
    log_string('----')
    num_batches_training = len(train_set) // (FLAGS.num_gpu * BATCH_SIZE)
    print('-----------------training--------------------')
    print('training steps: %d'%num_batches_training)
    total_loss = 0
    for i in tqdm(range(num_batches_training)):
        ###
        batch_train_data, batch_label_data = next(generator)
        # one = tf.ones_like(batch_label_data)
        # zero = tf.zeros_like(batch_label_data)
        # batch_mask_data = np.where(batch_label_data>0, 100, batch_label_data)
        # batch_mask_data = np.where(batch_mask_data == 0, 1, batch_mask_data)
        batch_mask_data = batch_train_data[:, :, :, 10].reshape(BATCH_SIZE, 80, 80, 1)
        batch_mask_data = np.where(batch_mask_data > 0, 10000, batch_mask_data)
        # batch_mask_data = np.where(batch_mask_data == 0, 1, batch_mask_data)
        ###
        feed_dict = {ops['input']: batch_train_data,
                     ops['label']: batch_label_data,
                     ops['mask']: batch_mask_data,
                     ops['is_training']: True,
                     }
        _, loss_, learning_rate_,step_, pre_ = sess.run([ops['train_op'], ops['loss'], ops['learning_rate'], ops['step'], ops['pre']], feed_dict=feed_dict)
        # _,learning_rate_ = sess.run([ops['train_op'], ops['learning_rate']], feed_dict=feed_dict)
        print('step:', step_)
        print('label:',batch_label_data.sum())
        print('pre:', pre_.sum())
        # print('learning rate:',learning_rate_)
        print('step loss:%f'%(loss_))
        total_loss += loss_
        if i%10 == 0:
            print('loss: %f'%(loss_))
    log_string('trianing_log_epoch_%d'%epoch)
    log_string('train_loss: %f'%(total_loss/(num_batches_training)))

def val_one_epoch(sess, epoch, val_set, generator, ops):
    log_string('----')
    num_batches_val = len(val_set) // (FLAGS.num_gpu * BATCH_SIZE)
    print('-----------------validation--------------------')
    total_loss = 0
    for _ in tqdm(range(num_batches_val)):
        ###
        batch_val_data, batch_label_data = next(generator)
        batch_mask_data = batch_val_data[:, :, :, 10].reshape(BATCH_SIZE, 80, 80, 1)
        batch_mask_data = np.where(batch_mask_data > 0, 10000, batch_mask_data)
        ###
        feed_dict = {ops['input']: batch_val_data,
                     ops['label']: batch_label_data,
                     ops['mask']: batch_mask_data,
                     ops['is_training']: False,
                     }
        loss_, pre_ = sess.run([ops['loss'], ops['pre']], feed_dict=feed_dict)
        total_loss += loss_

    log_string('val_log_epoch_%d' % epoch)
    log_string('val_loss: %f' % (total_loss / (num_batches_val)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()

