import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import model
import batchdata_generator_80 as DG
from pandas import DataFrame
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 2]')
parser.add_argument('--model_dir', default='train_log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training for each GPU [default: 12]')
parser.add_argument('--testing_data_path', default='./data/', help='Make sure the source training-data files path')
FLAGS = parser.parse_args()

TEST_DATA_PATH = FLAGS.testing_data_path
MODEL_DIR = FLAGS.model_dir
BATCH_SIZE = FLAGS.batch_size

NUM_R = 80
NUM_C = 80



def load_model():
    global ops
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            with tf.device('/gpu:0'):
                input = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 12))
                label = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                mask = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_R, NUM_C, 1))
                is_training = tf.compat.v1.placeholder(tf.bool, shape=())
                pre = model.Unet('unet', input, training=is_training)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=3)
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.compat.v1.Session(config=config)
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(MODEL_DIR))

        ops = {'input': input,
               'is_training': is_training,
               'label': label,
               'mask': mask,
               'pre': pre}

    return(sess, ops)


def estimate(sess, ops, lst_v, tg, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF):
    """ ops: dict mapping from string to tf ops """
    input = toGrid(lst_v, tg, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF)
    # print('-----------------testing--------------------')
    input = DG.normalize(input)
    input = input.reshape(1, 80, 80, 12)
    batch_mask_data = input[:, :, :, 10].reshape(1, 80, 80, 1)
    batch_mask_data = np.where(batch_mask_data > 0, 1, batch_mask_data)
    feed_dict = {ops['input']: input,
                 ops['mask']: batch_mask_data,
                 ops['is_training']: False}

        # print('-----label-----',label.sum())
        # print('___label___sum____', label.sum())
        # print('___error___sum___',(label-pre).sum())
        # print(label.sum())
        # print((pre-label).sum())
    pre_, mask_, input_ = sess.run([ops['pre'], ops['mask'], ops['input']], feed_dict=feed_dict)
    pre_ = pre_*batch_mask_data
    pre__ = pre_.flatten()
    return(pre__)

def toGrid(lst_v, tg, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF):
    result = np.empty([6400, 12], dtype=float)
    result[:, 0] = len(lst_v)

    result[:, 1] = tg

    lst_allTime, lst_nightTime, lst_dayTime, lst_peakTime, lst_allTimeH, lst_nightTimeH, lst_dayTimeH, lst_peakTimeH = toRegion(lst_v, df_vehInfo, df_f_r)

    df_ = DataFrame(lst_allTime, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 2] = lst

    df_ = DataFrame(lst_nightTime, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 3] = lst

    df_ = DataFrame(lst_dayTime, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 4] = lst

    df_ = DataFrame(lst_peakTime, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 5] = lst

    df_ = DataFrame(lst_allTimeH, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 6] = lst

    df_ = DataFrame(lst_nightTimeH, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 7] = lst

    df_ = DataFrame(lst_dayTimeH, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 8] = lst

    df_ = DataFrame(lst_peakTimeH, columns=['RegionP'])
    df_['FishnetID'] = df_.index
    df_['FishnetID'] = df_['FishnetID'] + 1
    df_ = df_RtoF.merge(df_, left_on='FishnetID', right_on='FishnetID', how='left')
    df_ = df_.fillna(value=0)
    df_ = df_.sort_values(by='FishIdOri', ascending=True)
    lst = list(df_['RegionP'])
    lst = np.array(lst)
    result[:, 9] = lst

    lst = list(df_popular['popularClass'])
    lst = np.array(lst)
    result[:, 10] = lst

    lst = list(df_bus['busStation_80_count'])
    lst = np.array(lst)
    result[:, 11] = lst

    return result

def toRegion(lst, df_vehInfo, df_f_r):
    df_tem = df_vehInfo[df_vehInfo['GA_ID'].isin(lst)]
    df_tem1 = df_tem.groupby(by=['Name_allTime'], as_index=False)['GA_ID'].count()
    df_tem1['allTime'] = df_tem1['GA_ID'] / len(df_tem)
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_allTime', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_allTime = list(df_tem1['allTime'])
    df_tem1 = df_tem.groupby(by=['Name_night'], as_index=False)['GA_ID'].count()
    df_tem1['nightTime'] = df_tem1['GA_ID'] / len(df_tem)
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_night', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_nightTime = list(df_tem1['nightTime'])
    df_tem1 = df_tem.groupby(by=['Name_day'], as_index=False)['GA_ID'].count()
    df_tem1['dayTime'] = df_tem1['GA_ID'] / len(df_tem)
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_day', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_dayTime = list(df_tem1['dayTime'])
    df_tem1 = df_tem.groupby(by=['Name_peak'], as_index=False)['GA_ID'].count()
    df_tem1['peakTime'] = df_tem1['GA_ID'] / len(df_tem)
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_peak', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_peakTime = list(df_tem1['peakTime'])
    df_tem1 = df_tem.groupby(by=['Name_allTime'], as_index=False)['all_h'].mean()
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_allTime', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_allTimeH = list(df_tem1['all_h'])
    df_tem1 = df_tem.groupby(by=['Name_night'], as_index=False)['nightTime_h'].mean()
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_night', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_nightTimeH = list(df_tem1['nightTime_h'])
    df_tem1 = df_tem.groupby(by=['Name_day'], as_index=False)['dayTime_h'].mean()
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_day', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_dayTimeH = list(df_tem1['dayTime_h'])
    df_tem1 = df_tem.groupby(by=['Name_peak'], as_index=False)['peakTime_h'].mean()
    df_tem1 = df_f_r.merge(df_tem1, left_on='NAME', right_on='Name_peak', how='left')
    df_tem1 = df_tem1.fillna(0)
    lst_peakTimeH = list(df_tem1['peakTime_h'])
    return lst_allTime, lst_nightTime, lst_dayTime, lst_peakTime, lst_allTimeH, lst_nightTimeH, lst_dayTimeH, lst_peakTimeH
