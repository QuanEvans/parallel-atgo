import os
from configure import script_dir

def evaluate_result_one(workdir, type, index,  round, data_type, method_name, number):

    # remove result dir
    resultdir = workdir + "/Predict_Label/" + type + "/Label/"
    os.system("rm -rf "+ resultdir)
    os.makedirs(resultdir)

    # copy results
    target_dir = workdir + "/" + type + "/resultset/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"
    os.system("cp " + target_dir + "/* " + resultdir + "/")

    test_gene_file = workdir + "/" + type + "/" + data_type + "_gene_list"
    test_label_file = workdir + "/" + type + "/" + data_type + "_gene_label"

    temp_result_dir = workdir + "/" + type + "/result/"
    os.system("rm -rf " + temp_result_dir)

    # create triplet results
    os.system("python2 " + script_dir + "/Run_pipelines_rank.py " + workdir + " " + test_gene_file + " " + str(number) + " " + type + " " + method_name)


    # copy triplet results
    triplet_result_dir = workdir+"/"+type+"/"+method_name+"_result/"+data_type+"/round"+str(round)+"/"
    if(os.path.exists(triplet_result_dir)==False):
        os.makedirs(triplet_result_dir)

    if(os.path.exists(triplet_result_dir + "/result" + str(index))):
        os.system("rm -rf " + triplet_result_dir + "/result" + str(index))

    os.system("cp -r "+ temp_result_dir +" " + triplet_result_dir + "/result" + str(index))


def evaluate_result(script_dir, workdir, type, index, round, number):


    method_name = "triplet"
    evaluate_result_one(script_dir, workdir, type, index, round, "test", method_name, number)













