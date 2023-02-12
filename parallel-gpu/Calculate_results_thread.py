import os
import numpy as np
from configure import script_dir, num_threads
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import closing
from sklearn.metrics import pairwise_distances
num_threads = min(num_threads, mp.cpu_count()//2)

# note: the maximum number of threads is half of the number of CPUs
# to avoid the nultiprocessing error
# and this step is I/O bound

def ed_distance(vec1, vec2): # calculate distance

    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def create_result_one_thread(args):
        resultdir, ed_distance_matrix_i, gene1, train_name, train_number = args
        resultJson = resultdir + f"{gene1}"

        if(os.path.exists(resultJson)):
            result_dict = dict()
            with open(resultJson, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        gene2, distance = line.split('\t')
                        result_dict[gene2] = float(distance)
        else:
            result_dict = dict()

        for j in range(train_number):

            gene2 = train_name[j]
            distance = ed_distance_matrix_i[j]

            if gene2 not in result_dict:
                result_dict[gene2] = distance
            else:
                result_dict[gene2] += distance

        with open(resultJson, 'w') as f:
            for gene2, distance in result_dict.items():
                f.write(f"{gene2}\t{distance}\n")

def create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, data_type, result_dict=None): # save results for triplet distance

    resultdir = workdir + "/" + type + "/resultset/" + data_type + "/round" + str(1) + "/result" + str(index) + "/"
    #os.system("rm -rf " + resultdir)
    # this step is modified to use only the frist round to store the results to save space
    
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    test_number = test_predict.shape[0]
    train_number = train_predict.shape[0]

    ed_distance_matrix =  pairwise_distances(test_predict, train_predict, metric='euclidean')

    args_list = []
    for i in range(test_number):
        gene1 = test_name[i]
        ed_distance_matrix_i = ed_distance_matrix[i]
        args_list.append([resultdir, ed_distance_matrix_i, gene1, train_name, train_number])
    
    with closing(Pool(num_threads)) as pool:
        pool.map(create_result_one_thread, args_list)
        pool.close()
        pool.join()


def calculate_result(workdir, train_predict, test_predict, train_name, test_name, index, type, round, result_dict=None): # save triplet distances for all datasets

    create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, "test",result_dict)

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


def calculate_pred_label_one_thread(args):
    resultdir, test_predict_matrix_i, term_list, name, method_name, type = args
    if (os.path.exists(resultdir + "/" + name + "/") == False):
        os.makedirs(resultdir + "/" + name)

    result_file = resultdir + "/" + name + "/" + method_name + "_" + type

    f = open(result_file, "w")

    for j in range(len(term_list)):

        if(test_predict_matrix_i[j]>=0.05):
            f.write(term_list[j] + " " + type[1:] + " " + str(test_predict_matrix_i[j]) + "\n")
    f.flush()
    f.close()


def calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, data_type, method_name):  # save cross entropy results

    term_file = workdir + "/" + type + "/term_list"
    term_list = read_term_list(term_file)

    resultdir = workdir + "/" + type + "/result/"
    os.system("rm -rf "+resultdir)
    os.makedirs(resultdir)

    args_list = []
    for i in range(len(test_name_list)):
        name = test_name_list[i]
        test_predict_matrix_i = test_predict_matrix[i]
        args_list.append([resultdir, test_predict_matrix_i, term_list, name, method_name, type])

    with closing(Pool(num_threads)) as pool:
        pool.map(calculate_pred_label_one_thread, args_list)
        pool.close()
        pool.join()

    post_deal1(workdir, type, method_name, data_type, index, round)

def calculate_pred_label(workdir, type, test_name_list, test_predict_matrix, round, index):  # save cross entropy results process

    method_name = "cross_entropy"

    calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, "test", method_name)



















