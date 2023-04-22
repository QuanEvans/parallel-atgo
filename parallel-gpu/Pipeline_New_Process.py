import sys
import os

from decimal import Decimal
import Find_Parents as fp
from configure import database_dir, blast_file, go_term_dir,num_threads
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import closing

num_threads = min(num_threads, mp.cpu_count()-4)
# note: the maximum number of threads is half of the number of CPUs
# to avoid the nultiprocessing error


def read_go(gofile):  # read GO Terms

    f = open(gofile, "r")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict


# golabl variables for multiprocessing
obo_dict = fp.get_obo_dict()
go_dict_aspects = {}
aspects = ["MF", "BP", "CC"]
for aspect in aspects:
    go_term_file = go_term_dir + "/" + aspect + "_Term"
    go_dict = read_go(go_term_file)
    go_dict_aspects[aspect] = go_dict


def read_protein_list(protein_list_file):    # read protein templates

    f = open(protein_list_file, "r")
    text = f.read()
    f.close()

    protein_list_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        protein_list_dict[values[0]] = float(values[1])

    return protein_list_dict

def annotate(args):  # annotate GO term

    workdir, type = args[0], args[1]
    go_dict = go_dict_aspects[type]

    protein_list_file = workdir+"/" + type + "_protein_list"
    if(os.path.exists(protein_list_file)==False or os.path.getsize(protein_list_file)==0):
        print("protein list is not exist ! ")
        return

    protein_list_dict = read_protein_list(protein_list_file)

    term_list = []
    for protein in protein_list_dict:
        term_list.extend(go_dict[protein])
    term_list = list(set(term_list))

    result_dict = dict()
    for term in term_list:
        sum1 = 0.0
        sum2 = 0.0
        for protein in protein_list_dict:
            sum1 = sum1 + protein_list_dict[protein]
            if(term in go_dict[protein]):
                sum2 = sum2 + protein_list_dict[protein]

        result_dict[term] = sum2/sum1

    result_list = [(result_dict[term], term) for term in result_dict]
    result_list = sorted(result_list, reverse=True)

    resultfile = workdir + "/protein_Result_" + type
    f = open(resultfile, "w")
    for value, term in result_list:
        if(value>=0.01):
            f.write(term+" "+type[1]+" "+str(Decimal(value).quantize(Decimal("0.000"))) + "\n")
    f.flush()
    f.close()

    fp.find_parents_from_file(resultfile, resultfile+"_new")

def process(args):   # main process
    workdir = args[0]
    aspect = args[1]
    annotate([workdir, aspect])


if __name__ == '__main__':

    workdir = sys.argv[1]
    aspects = ["MF", "BP", "CC"]

    input_name = []
    for name in os.listdir(workdir):
        for i in range(len(aspects)):
            aspect = aspects[i]
            input_name.append([workdir + "/" + name + "/", aspect])

    with closing(Pool(processes=num_threads)) as pool:
        pool.map(process, input_name)
        pool.close()
        pool.join()









