import os
import sys
from decimal import Decimal
import Find_Parents as fp
from configure import script_dir, num_threads
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import closing
num_threads = min(num_threads, mp.cpu_count()-4)

def read_gene_go_dict(gene_term_map_file):  # read go dict

    f = open(gene_term_map_file, "r")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict

def read_gene_list(gene_list_file):  # read gene list

    if(os.path.exists(gene_list_file) == False):
        return []

    f = open(gene_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def read_source_file(source_file, go_dict, number, start_index): # read resource

    f = open(source_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()[start_index:]

    count = 0

    gene_term_dict = dict()
    gene_value_dict = dict()
    go_term_list = []

    for line in line_set:

        if(count>=number):
            break
        line = line.strip()
        values = line.split()
        gene_id = values[0]
        co_expression_value = (float(number-count))/number

        if(gene_id in go_dict):

            gene_term_dict[gene_id] = go_dict[gene_id]
            gene_value_dict[gene_id] = co_expression_value
            go_term_list.extend(go_dict[gene_id])
            count = count + 1

    go_term_list = list(set(go_term_list))

    return gene_term_dict, gene_value_dict, go_term_list

def calculate_single_result(source_file, result_file, go_dict, number, obo_dict, type, start_index): # create single result

    gene_term_dict, gene_value_dict, go_term_list = read_source_file(source_file, go_dict, number, start_index)

    result_dict = dict()

    for term in go_term_list:

        sum_weight = 0.0
        sum = 0.0

        for gene in gene_value_dict:
            if(term in gene_term_dict[gene]):
                sum = sum + float(gene_value_dict[gene])
            sum_weight = sum_weight + float(gene_value_dict[gene])

        result_dict[term] = sum/sum_weight

    result_list = [(result_dict[term], term) for term in result_dict]
    result_list = sorted(result_list, reverse=True)

    f = open(result_file, "w")
    for value, gene in result_list:
        if(value>=0.01):
            f.write(gene + " "+ type[1:] + " " + str(Decimal(value).quantize(Decimal("0.000"))) + "\n")
    f.flush()
    f.close()

    fp.find_parents_from_file(result_file, result_file + "_new")

def calculate_single_result_mt(args): # create single result
    source_file, result_file, number, type, start_index = args
    go_dict = go_dict_aspect[type]
    
    gene_term_dict, gene_value_dict, go_term_list = read_source_file(source_file, go_dict, number, start_index)

    result_dict = dict()

    for term in go_term_list:

        sum_weight = 0.0
        sum = 0.0

        for gene in gene_value_dict:
            if(term in gene_term_dict[gene]):
                sum = sum + float(gene_value_dict[gene])
            sum_weight = sum_weight + float(gene_value_dict[gene])

        result_dict[term] = sum/sum_weight

    result_list = [(result_dict[term], term) for term in result_dict]
    result_list = sorted(result_list, reverse=True)

    f = open(result_file, "w")
    for value, gene in result_list:
        if(value>=0.01):
            f.write(gene + " "+ type[1:] + " " + str(Decimal(value).quantize(Decimal("0.000"))) + "\n")
    f.flush()
    f.close()

    fp.find_parents_from_file(result_file, result_file + "_new")

def create_single_pipeline_result(gene_list, source_dir, pipeline, type, resultdir, number, start_index):
    # this is changed to multi-threading
    args_list = []

    for gene in gene_list:

        source_file = source_dir + "/" + gene
        if (os.path.exists(resultdir + "/" + gene + "/") == False):
            os.makedirs(resultdir + "/" + gene + "/")

        result_file = resultdir + "/" + gene + "/" + pipeline + "_" + type

        args_list.append((source_file, result_file, number, type, start_index))
    
    with closing(Pool(processes=num_threads)) as pool:
        pool.map(calculate_single_result_mt, args_list)
        pool.close()
        pool.join()
        #calculate_single_result(source_file, result_file, go_dict, number, obo_dict, type, start_index)


def process(dir, test_gene_file, number, type, pipe_type):

    type_list = [type]
    pipeline_list = [pipe_type]
    global obo_dict
    obo_dict = fp.get_obo_dict()

    global go_dict_aspect
    go_dict_aspect = {}

    for type in type_list:


        source_list = ["/Predict_Label/" + type + "/Label/"]
        start_index_list = [0]
        gene_list = read_gene_list(test_gene_file)

        go_map_file = dir + "/" + type + "/gene_GO_Terms"
        go_dict = read_gene_go_dict(go_map_file)
        go_dict_aspect[type] = go_dict

        result_dir = dir + "/" + type + "/result/"

        for i in range(len(pipeline_list)):

            pipeline = pipeline_list[i]
            source_dir = dir + "/" + source_list[i] + "/"
            start_index = start_index_list[i]

            create_single_pipeline_result(gene_list, source_dir, pipeline, type, result_dir, number, start_index)

if __name__=="__main__":

    process(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])








