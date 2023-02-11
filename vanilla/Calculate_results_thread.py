import os
import numpy as np

from configure import script_dir

def ed_distance(vec1, vec2): # calculate distance

    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, data_type): # save results for triplet distance

    resultdir = workdir + "/" + type + "/resultset/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"
    os.system("rm -rf " + resultdir)
    os.makedirs(resultdir)

    test_number = test_predict.shape[0]
    train_number = train_predict.shape[0]

    for i in range(test_number):

        gene1 = test_name[i]
        vector1 = test_predict[i]

        result_list = []

        for j in range(train_number):

            gene2 = train_name[j]
            vector2 = train_predict[j]

            distance = ed_distance(vector1, vector2)
            result_list.append((distance, gene2))

        result_list = sorted(result_list)

        resultfile = resultdir + "/" + gene1
        f = open(resultfile, "w")
        for value, name in result_list:
            f.write(name + " " + str(value) + "\n")
        f.flush()
        f.close()


def calculate_result(workdir, train_predict, test_predict, train_name, test_name, index, type, round): # save triplet distances for all datasets

    create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, "test")

def read_term_list(term_file):  # read term_list

    f = open(term_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def post_deal1(workdir, type, method_name, data_type, index, round): # evaluate results and save results

    resultdir = workdir + "/" + type + "/result/"

    os.system("python2 " + script_dir + "/Find_Parents.py " + resultdir + " " + type + " " + method_name)

    copy_result_dir = workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/round" + str(round) + "/"
    if (os.path.exists(copy_result_dir ) == False):
        os.makedirs(copy_result_dir )

    if (os.path.exists(copy_result_dir  + "/result" + str(index))):
        os.system("rm -rf " + copy_result_dir  + "/result" + str(index))

    os.system("cp -r " + resultdir + " " + copy_result_dir  + "/result" + str(index))

def calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, data_type, method_name):  # save cross entropy results

    term_file = workdir + "/" + type + "/term_list"
    term_list = read_term_list(term_file)

    resultdir = workdir + "/" + type + "/result/"
    os.system("rm -rf "+resultdir)
    os.makedirs(resultdir)

    for i in range(len(test_name_list)):

        name = test_name_list[i]

        if(os.path.exists(resultdir + "/" + name + "/")==False):
            os.makedirs(resultdir + "/" + name)

        result_file = resultdir + "/" + name + "/" + method_name + "_" + type

        f = open(result_file, "w")

        for j in range(len(term_list)):

            if(test_predict_matrix[i][j]>=0.05):
                f.write(term_list[j] + " " + type[1:] + " " + str(test_predict_matrix[i][j]) + "\n")
        f.flush()
        f.close()

    post_deal1(workdir, type, method_name, data_type, index, round)

def calculate_pred_label(workdir, type, test_name_list, test_predict_matrix, round, index):  # save cross entropy results process

    method_name = "cross_entropy"

    calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, "test", method_name)



















