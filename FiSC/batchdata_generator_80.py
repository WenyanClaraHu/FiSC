#coding=utf-8
import random
import numpy as np
import os

def load_data(path):
    try:
        return np.load(path)
    except:
        return np.loadtxt(path)

def get_train_val_set(trainingdata_path, val_rate=0.20):
    train_set = []
    val_set = []
    all_train_set = os.listdir(trainingdata_path)
    random.shuffle(all_train_set)
    total_num = len(all_train_set)
    val_num = int(val_rate * total_num)
    for j in range(len(all_train_set)):
        if j < val_num:
            val_set.append(all_train_set[j])
        else:
            train_set.append(all_train_set[j])
    return train_set, val_set

def normalize(temp_input):
    max_input = np.array([1000.0, 12.0, 1.0, 1.0, 1.0, 1.0, 24.0, 12.0, 8.0, 4.0, 19.0, 47.0], dtype = np.float32)
    min_input = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype = np.float32)
    # max_input = np.array([1000.0, 12.0, 1.0, 9068.0, 2879.0], dtype = np.float32)
    # min_input = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype = np.float32)
    # min_input = np.min(temp_input, axis=0)
    # max_input = np.max(temp_input, axis=0)
    nor_input = (temp_input - min_input) / (max_input - min_input)
    return nor_input

def generate_training_minibatch_data(trainingdata_path, batch_size, train_set):
    while True:
        train_input_data = []
        label_data = []
        batch = 0
        for i in (range(len(train_set))):
            batch += 1
            url = train_set[i]
            temp_input_set = load_data(trainingdata_path+url)
            #####
            temp_input = temp_input_set[:, :12]
            temp_input = normalize(temp_input)
            temp_input = temp_input.reshape((80, 80, 12))
            # temp_input = temp_input.reshape((160, 160, 5))

            label = temp_input_set[:, -1]
            label = label.reshape((80, 80, 1))

            train_input_data.append(temp_input)
            label_data.append(label)

            if batch % batch_size == 0:
                train_input_data = np.array(train_input_data, dtype = np.float32)
                label_data = np.array(label_data, dtype = np.float32)
                yield [train_input_data, label_data]
                train_input_data = []
                label_data = []
                batch = 0


if __name__ == "__main__":

    datapath = './data/'
    train_set, val_set = get_train_val_set(datapath)
    generator = generate_training_minibatch_data(datapath, 1, train_set)

    for _ in range(5):
        train_data, vote_label = next(generator)
        train_data = np.squeeze(train_data)
        vote_lable = np.squeeze(vote_label)
        temp_output = np.concatenate([train_data, np.expand_dims(vote_label, -1)], axis=-1)
        temp_output = np.squeeze(temp_output)
        np.savetxt('./temp/out.txt', temp_output, fmt='%.4f')


    aa = 1




