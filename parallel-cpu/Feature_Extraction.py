import sys
import os
import Create_Features_Combine as cf
from configure import python_dir, esm_model_name, script_dir, data_dir


def run_esm(workdir):

    feature_embeddings_dir = workdir + "/feature_embeddings/mean/"
    os.system("rm -rf " + feature_embeddings_dir)
    os.makedirs(feature_embeddings_dir)

    cmd = python_dir + " " + script_dir + "/extract.py " + esm_model_name + " " + workdir + "/seq.fasta " + feature_embeddings_dir + " --repr_layers 31 32 33 --include mean --truncate"
    os.system(cmd)

    cmd = python_dir + " " + script_dir + "/extract_single_feature.py " + workdir
    os.system(cmd)

    combine_features(workdir)
    create_test_files(workdir)

def combine_features(workdir):

    feature_dir1 = workdir + "/feature_embeddings/31/"
    feature_dir2 = workdir + "/feature_embeddings/32/"
    feature_dir3 = workdir + "/feature_embeddings/33/"

    name_list = workdir + "/name_list"
    feature_file = workdir + "/test_feature"
    cf.create_feature(feature_dir1, feature_dir2, feature_dir3, name_list, feature_file)



def create_test_files(workdir):

    type_list = ["MF", "BP", "CC"]
    term_number_dict = dict()
    term_number_dict["MF"] = 6581
    term_number_dict["BP"] = 4133
    term_number_dict["CC"] = 2782

    root_dict = dict()
    root_dict["MF"] = "GO:0003674"
    root_dict["BP"] = "GO:0008150"
    root_dict["CC"] = "GO:0005575"

    f = open(workdir + "/name_list", "r")
    text = f.read()
    f.close()
    name_list = text.splitlines()

    for type in type_list:

        tempdir = workdir + "/ATGO/" + type + "/"
        os.system("rm -rf " + tempdir)
        os.makedirs(tempdir)

        os.system("cp " + workdir + "/test_feature " + tempdir + "/test_feature")

        f = open(tempdir + "/test_gene_list", "w")
        for name in name_list:
            f.write(name + "\n")
        f.flush()
        f.close()

        f = open(tempdir + "/test_label_one_hot", "w")
        for name in name_list:
            line = ""
            for i in range(term_number_dict[type]):
                line = line + "0 "
            f.write(line + "\n")

        f.flush()
        f.close()

        f = open(tempdir + "/test_gene_label", "w")
        for name in name_list:
            f.write(name + "  " + root_dict[type] + "\n")
        f.flush()
        f.close()

        os.system("cp " + data_dir + "/" + type + "/train_gene_label " + tempdir + "/")
        os.system("cp " + data_dir + "/" + type + "/gene_GO_Terms " + tempdir + "/")
        os.system("cp " + data_dir + "/" + type + "/term_list " + tempdir + "/")


if __name__=="__main__":

    workdir = sys.argv[1]
    run_esm(workdir)



