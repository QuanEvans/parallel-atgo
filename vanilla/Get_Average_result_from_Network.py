import os
import sys
import Create_Average_Result as car
import Evaluate_Improve_New as ei

from configure import script_dir

def get_average_result_from_triplet_network(workdir, type, index, times, select_number):  # get triplet results

    car.create_average_result(workdir + "/" + type + "/", index, times)
    ei.evaluate_result(workdir, type, index, select_number)

def read_result(result_file):   # read results

    f = open(result_file, "r")
    text = f.read()
    f.close()

    result_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        result_dict[values[0]] = float(values[2])

    return result_dict

def post_deal1(workdir, type, method_name, data_type, index): # evaluate results and save results

    resultdir = workdir + "/" + type + "/result/"

    os.system("python2 " + script_dir + "/Find_Parents.py " + resultdir + " " + type + " " + method_name)

    copy_result_dir = workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/"
    if (os.path.exists(copy_result_dir) == False):
        os.makedirs(copy_result_dir)

    if (os.path.exists(copy_result_dir  + "/result" + str(index))):
        os.system("rm -rf "+copy_result_dir  + "/result" + str(index))

    os.system("cp -r " + resultdir + " " + copy_result_dir  + "/result" + str(index))


def create_single_result_one(workdir, type, index, times, data_type, method_name):  # create cross entropy results

    resultdir = workdir + "/" + type + "/result/"
    os.system("rm -rf "+resultdir)

    name_list = sorted(os.listdir(workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/round1/result" + str(index) + "/"))

    for name in name_list:

        final_result_dict = dict()

        for i in range(1, times+1):

            result_file = workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/round" + str(i) + "/result" + str(index) + "/" + name + "/" +method_name + "_" + type
            result_dict = read_result(result_file)

            for term in result_dict:
                if(term not in final_result_dict):
                    final_result_dict[term] = result_dict[term]
                else:
                    final_result_dict[term] = final_result_dict[term] + result_dict[term]

        os.makedirs(resultdir + name + "/")

        f = open(resultdir + name + "/final_" + method_name + "_" + type, "w")


        for term in final_result_dict:

            final_result_dict[term] = final_result_dict[term]/times

            if(final_result_dict[term]>=0.01):
                f.write(term + " " + type[1] + " " + str(final_result_dict[term]) + "\n")

        f.flush()
        f.close()

    post_deal1(workdir, type, "final_"+method_name, data_type, index)


def create_single_result(workdir, type, index, times):  # # create cross entropy results process

    method_name = "cross_entropy"
    create_single_result_one(workdir, type, index, times, "test", method_name)

def read_result_by_name(file): # read results

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


def combine_one(workdir, type, index, weight, method_name1, method_name2, method_name3, data_type):

    resultdir1 = workdir + "/" + type + "/final_" + method_name1 + "_result/" + data_type + "/result" + str(index) + "/"
    resultdir2 = workdir + "/" + type + "/final_" + method_name2 + "_result/" + data_type + "/result" + str(index) + "/"

    resultdir3 = workdir+"/"+type+"/result/"
    os.system("rm -rf "+resultdir3)
    os.makedirs(resultdir3)

    name_list = os.listdir(resultdir1)


    for name in name_list:

        file1 = resultdir1+"/"+name+"/final_" + method_name1 + "_" + type
        file2 = resultdir2+"/"+name+"/final_" + method_name2 + "_" + type
        file3 = resultdir3+"/"+name+"/final_" + method_name3 + "_" + type

        if(os.path.exists(resultdir3+"/"+name+"/")==False):
            os.makedirs(resultdir3+"/"+name+"/")

        combine_result(file1, file2, file3, weight, type)

    post_deal1(workdir, type, "final_" + method_name3, data_type, index)

def combine(workdir, type, index, weight):


    method_name1 = "triplet"
    method_name2 = "cross_entropy"
    method_name3 = "combine"
    combine_one(workdir, type, index, weight, method_name1, method_name2, method_name3, "test")



def process(workdir, type, index, weight, times, select_number):


    get_average_result_from_triplet_network(workdir, type, index, times, select_number)
    create_single_result(workdir, type, index, times)
    combine(workdir, type, index, weight)


if __name__=="__main__":

    workdir = sys.argv[1]

    index_dict = dict()
    index_dict["MF"] = 630
    index_dict["BP"] = 580
    index_dict["CC"] = 270

    weight_dict = dict()
    weight_dict["MF"] = 0.1
    weight_dict["BP"] = 0.4
    weight_dict["CC"] = 0.2

    number_dict = dict()
    number_dict["MF"] = 30
    number_dict["BP"] = 100
    number_dict["CC"] = 100

    times = 10

    type_list = ["MF", "BP", "CC"]
    for type in type_list:
        process(workdir, type, index_dict[type], weight_dict[type], times, number_dict[type])





