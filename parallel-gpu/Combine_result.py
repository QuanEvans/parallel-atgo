import sys
import os
import Find_Parents as fp


def read_name_list(name_list_file):

    f = open(name_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def read_result_by_name(file): # read results

    if(os.path.exists(file)==False):
        return [], []

    f = open(file, "r")
    text = f.read()
    f.close()
    name_list = []

    result_dict = dict()
    for line in text.splitlines():
        values = line.strip().split()
        result_dict[values[0]] = float(values[2])
        name_list.append(values[0])

    return result_dict, name_list

def combine_result(file1, file2, file3, weight, type):  # combine results

    result_dict1, name_list1 = read_result_by_name(file1)
    result_dict2, name_list2 = read_result_by_name(file2)

    result_dict3 = dict()

    name_list1.extend(name_list2)
    name_list= list(set(name_list1))

    for name in name_list:

        if (name in result_dict1):
            value1 = result_dict1[name]
        else:
            value1 = 0

        if(name in result_dict2):
            value2 = result_dict2[name]
        else:
            value2 = 0

        value = weight * value1 + (1-weight) * value2
        result_dict3[name] = value

    f = open(file3, "w")
    for name in result_dict3:
        if(result_dict3[name]>=0.01):
            f.write(name + " " + type[1:] + " " + str(result_dict3[name]) + "\n")
    f.flush()
    f.close()

def copy_triplet_result(workdir):

    type_list = ["MF", "BP", "CC"]
    index_dict = dict()
    index_dict["MF"] = 630
    index_dict["BP"] = 580
    index_dict["CC"] = 270

    for type in type_list:
        atgo_dir = workdir + "/ATGO/" + type + "/final_combine_result/test/result" + str(index_dict[type]) + "/"
        name_list = os.listdir(atgo_dir)

        for name in name_list:
            originfile = atgo_dir + "/" + name + "/final_combine_" + type + "_new"
            copydir = workdir + "/ATGO_PLUS/" + name + "/"
            if(os.path.exists(copydir)==False):
                os.makedirs(copydir)
            copyfile = copydir + "/ATGO_" + type
            os.system("mv " + originfile + " " + copyfile)



def copy_sagp_result(workdir):

    type_list = ["MF", "BP", "CC"]

    weight_dict = dict()
    weight_dict["MF"] = 0.57
    weight_dict["BP"] = 0.60
    weight_dict["CC"] = 0.67

    obo_dict = fp.get_obo_dict()


    sagp_dir = workdir + "/SAGP/"
    name_list = os.listdir(sagp_dir)
    for name in name_list:
        for type in type_list:
            originfile = sagp_dir + "/" + name + "/protein_Result_" + type + "_new"
            copydir = workdir + "/ATGO_PLUS/" + name + "/"
            if (os.path.exists(copydir) == False):
                os.makedirs(copydir)
            copyfile = copydir + "/SAGP_" + type

            if(os.path.exists(originfile)):

                os.system("mv " + originfile + " " + copyfile)

            file1 = copydir + "/ATGO_" + type
            file2 = copydir + "/SAGP_" + type
            file3 = copydir + "/ATGO_PLUS_" + type
            combine_result(file1, file2, file3, weight_dict[type], type)

            fp.find_parents_from_file(file3, file3, obo_dict)
            fp.sort_result(file3)

        os.system("mv " + sagp_dir + "/" + name + "/seq.fasta " + copydir + "/seq.txt")




def process(workdir):

    copy_triplet_result(workdir)
    copy_sagp_result(workdir)






if __name__=="__main__":

    workdir = sys.argv[1]
    process(workdir)



