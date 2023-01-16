# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.random.set_seed(0)

def Unet(name, in_data, training, reuse=False):
    # Not use BatchNorm or InstanceNorm.
    assert in_data is not None
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # Conv1
        conv1_1 = tf.compat.v1.layers.conv2d(in_data, 64, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer=tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv1_2 = tf.compat.v1.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())
            # kernel_initializer=tf.contrib.layers.xavier_initializer())

        # MaxPooling1 80
        pool1 = tf.compat.v1.layers.max_pooling2d(conv1_2, 2, 2)
        conv2_1 = tf.compat.v1.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.compat.v1.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())
            # kernel_initializer = tf.contrib.layers.xavier_initializer())

        # MaxPooling2 40
        pool2 = tf.compat.v1.layers.max_pooling2d(conv2_2, 2, 2)
        conv3_1 = tf.compat.v1.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.compat.v1.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())
            # kernel_initializer = tf.contrib.layers.xavier_initializer())

        ####20
        pool3 = tf.compat.v1.layers.max_pooling2d(conv3_2, 2, 2)
        ############
        conv4_1 = tf.compat.v1.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv4_2 = tf.compat.v1.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        drop4 = tf.compat.v1.layers.dropout(conv4_2, training=training)
       #
        # MaxPooling4 10
        pool4 = tf.compat.v1.layers.max_pooling2d(drop4, 2, 2)
        conv5_1 = tf.compat.v1.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv5_2 = tf.compat.v1.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        drop5 = tf.compat.v1.layers.dropout(conv5_2,  training=training)

        # Upsampling6  20
        up6_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)
        up6 = tf.compat.v1.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        merge6 = tf.concat([conv4_2, up6], axis=3) # concat
        # Conv6 + Upsampling7 + Conv + Merge7
        conv6_1 = tf.compat.v1.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv6_2 = tf.compat.v1.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())

        ####40
        up7_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        up7 = tf.compat.v1.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        merge7 = tf.concat([conv3_2, up7], axis=3) # concat channel
        # Conv7 + Upsampling8 + Conv + Merge8
        conv7_1 = tf.compat.v1.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv7_2 = tf.compat.v1.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())


        ####80
        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        up8 = tf.compat.v1.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        merge8 = tf.concat([conv2_2, up8], axis=3) # concat channel
        # Conv8 + Upsampling9 + Conv + Merge9
        conv8_1 = tf.compat.v1.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv8_2 = tf.compat.v1.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())

        ####160
        up9_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        up9 = tf.compat.v1.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        merge9 = tf.concat([conv1_2, up9], axis=3)
        # Conv9
        conv9_1 = tf.compat.v1.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv9_2 = tf.compat.v1.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu, padding='same',
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv9_3 = tf.compat.v1.layers.conv2d(conv9_2, 16, 3, padding="SAME", activation=tf.nn.relu,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
        conv10 = tf.compat.v1.layers.conv2d(conv9_3, 1, 1, activation=tf.nn.sigmoid,
            # kernel_initializer = tf.contrib.layers.xavier_initializer())
            kernel_initializer=tf.keras.initializers.glorot_uniform())
    return conv10



def get_loss(pre, label, mask):
    loss = tf.reduce_sum(tf.reduce_mean(tf.square((label - pre)*mask), axis=0)) / 3420.0
    return loss
