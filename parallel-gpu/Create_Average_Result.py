import sys
import os
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import closing
from configure import  num_threads
num_threads = min(num_threads, mp.cpu_count()-4)

def read_result(result_file):  # read result file

    f = open(result_file, "r")
    text = f.read()
    f.close()

    result_dict = dict()

    rank = 1

    for line in text.splitlines():

        values = line.strip().split()
        result_dict[values[0]] = float(values[1])
        rank = rank + 1

    return result_dict

def create_single_average_result(workdir, index, times, name, type):

    result_file = workdir + "/resultset/test/round1/result" + str(index) + "/" + name

    with open(result_file, 'r') as result:
        final_result_dict = dict()
        for line in result:
            line = line.strip()
            if line:
                gene2, distance = line.split('\t')
                final_result_dict[gene2] = float(distance)

    for gene in final_result_dict:
        final_result_dict[gene] = final_result_dict[gene]/times

    final_result_list = [(final_result_dict[gene], gene) for gene in final_result_dict]
    final_result_list = sorted(final_result_list)

    result_file = workdir + "/result_set/" + type + "/result" + str(index) + "/" + name
    f = open(result_file, "w")
    for value, gene in final_result_list:
        f.write(gene+" "+str(value)+"\n")
    f.flush()
    f.close()

def create_single_average_result_mt(args):
    workdir, index, times, name, type = args
    create_single_average_result(workdir, index, times, name, type)


def create_average_result(workdir, index, times):  # create average result

    name_list = sorted(os.listdir(workdir + "/resultset/test/round1/result" + str(index) + "/"))

    os.system("rm -rf " + workdir + "/result_set/test/result" + str(index) + "/")
    os.makedirs(workdir + "/result_set/test/result" + str(index) + "/")

    for name in name_list:
        create_single_average_result(workdir, index, times, name, "test")

def create_average_result_mt(workdir, index, times):  # create average result

    name_list = sorted(os.listdir(workdir + "/resultset/test/round1/result" + str(index) + "/"))

    os.system("rm -rf " + workdir + "/result_set/test/result" + str(index) + "/")
    os.makedirs(workdir + "/result_set/test/result" + str(index) + "/")

    args = [(workdir, index, times, name, "test") for name in name_list]
    with closing(Pool(num_threads)) as pool:
        pool.map(create_single_average_result_mt, args)


if __name__=="__main__":

    #create_average_result(sys.argv[1])
    create_average_result_mt(sys.argv[1])
    





