import numpy as np
import sys
import random
import math

def read_name_list(name_file):   # read name file

    f = open(name_file, "r")
    text = f.read()
    f.close()

    name_list = []
    for line in text.splitlines():
        name_list.append(line.strip().split()[0])

    return np.array(name_list)

def read_train_evaluate_data(feature_dir, type):  # read feature, label, name

    train_feature_file = feature_dir + "/" + type + "/train_feature"
    train_feature = np.loadtxt(train_feature_file)

    train_label_file = feature_dir + "/" + type + "/train_label_one_hot"
    train_label = np.loadtxt(train_label_file)


    train_name_file = feature_dir+"/"+type+"/train_gene_label"
    train_name = read_name_list(train_name_file)

    print(train_feature.shape)
    print(train_label.shape)
    print(len(train_name))

    return train_feature, train_label, train_name



def read_test_data(feature_dir, type):

    test_feature_file = feature_dir + "/" + type + "/test_feature"
    test_feature = np.matrix(np.loadtxt(test_feature_file))

    test_label_file = feature_dir + "/" + type + "/test_label_one_hot"
    test_label = np.matrix(np.loadtxt(test_label_file))

    test_name_file = feature_dir + "/" + type + "/test_gene_label"
    test_name = read_name_list(test_name_file)

    print(test_feature.shape)
    print(test_label.shape)
    print(len(test_name))


    return test_feature, test_label, test_name



def create_batch(feature, label, name, batch_size, is_shuffle=True):  #create batch

    number = len(name)
    index = [i for i in range(number)]
    
    if(is_shuffle):
        random.shuffle(index)

    batch_number = math.ceil(float(number)/batch_size)

    data_list = []
    for i in range(batch_number):

        start = i*batch_size
        end = (i+1)*batch_size
        if(end>number):
            end = number

        current_index = sorted(index[start:end])

        current_feature = feature[current_index]
        current_label = label[current_index]
        current_name = name[current_index]

        data_list.append((current_feature, current_label,current_name))

    return data_list

