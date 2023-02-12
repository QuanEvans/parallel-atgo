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

def read_data(feature_dir, type):  # read feature, label, name

    train_feature_file = feature_dir + "/" + type + "/train_feature"
    train_feature = np.loadtxt(train_feature_file)

    evaluate_feature_file = feature_dir + "/" + type + "/evaluate_feature"
    evaluate_feature = np.loadtxt(evaluate_feature_file)

    test_feature_file = feature_dir + "/" + type + "/test_feature"
    test_feature = np.loadtxt(test_feature_file)


    train_label_file = feature_dir + "/" + type + "/train_label_one_hot"
    train_label = np.loadtxt(train_label_file)

    evaluate_label_file = feature_dir + "/" + type + "/evaluate_label_one_hot"
    evaluate_label = np.loadtxt(evaluate_label_file)

    test_label_file = feature_dir + "/" + type + "/test_label_one_hot"
    test_label = np.loadtxt(test_label_file)


    train_name_file = feature_dir+"/"+type+"/train_gene_label"
    train_name = read_name_list(train_name_file)

    evaluate_name_file = feature_dir + "/" + type + "/evaluate_gene_label"
    evaluate_name = read_name_list(evaluate_name_file)

    test_name_file = feature_dir + "/" + type + "/test_gene_label"
    test_name = read_name_list(test_name_file)


    print(train_feature)
    print(train_feature.shape)

    print(train_label)
    print(train_label.shape)

    print(train_name)
    print(train_name.shape)

    print(evaluate_feature)
    print(evaluate_feature.shape)

    print(evaluate_label)
    print(evaluate_label.shape)

    print(evaluate_name)
    print(evaluate_name.shape)

    print(test_feature)
    print(test_feature.shape)

    print(test_label)
    print(test_label.shape)

    print(test_name)
    print(test_name.shape)


    return train_feature, train_label, train_name, evaluate_feature, evaluate_label, evaluate_name, test_feature, test_label, test_name

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

