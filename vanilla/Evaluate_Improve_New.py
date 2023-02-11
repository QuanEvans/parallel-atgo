import os
import sys
from configure import script_dir

def evaluate_result_one(workdir, type, index, data_type, method_name, number):

    # create result dir
    resultdir = workdir + "/Predict_Label/" + type + "/Label/"
    os.system("rm -rf " + resultdir)
    os.makedirs(resultdir)

    # copy files
    target_dir = workdir + "/" + type + "/result_set/" + data_type + "/result" + str(index) + "/"
    os.system("cp " + target_dir + "/* " + resultdir + "/")

    test_gene_file = workdir + "/" + type + "/" + data_type + "_gene_list"


    temp_result_dir = workdir+"/"+type+"/result/"
    os.system("rm -rf "+ temp_result_dir)

    os.system("python2 " + script_dir + "/Run_pipelines_rank.py " + workdir + " " + test_gene_file + " " + str(number) + " " + type+" "+ method_name)

    copy_result_dir = workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/"
    if (os.path.exists(copy_result_dir) == False):
        os.makedirs(copy_result_dir)

    if (os.path.exists(copy_result_dir + "/result" + str(index))):
        os.system("rm -rf "+copy_result_dir + "/result" + str(index))

    os.system("cp -r " + temp_result_dir + " " + copy_result_dir + "/result" + str(index))

def evaluate_result(workdir, type, index, number):

    method_name = "final_triplet"
    evaluate_result_one(workdir, type, index, "test", method_name, number)










