
import sys
import os

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

    result_list = []

    for i in range(1, times+1):

        result_file = workdir + "/resultset/" + type + "/round" + str(i) + "/result" + str(index) + "/" + name
        result_dict = read_result(result_file)
        result_list.append(result_dict)

    final_result_dict = dict()

    for gene in result_list[0]:

        for i in range(times):

            if(gene not in final_result_dict):

                final_result_dict[gene] = result_list[i][gene]
            else:
                final_result_dict[gene] = final_result_dict[gene] + result_list[i][gene]

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


def create_average_result(workdir, index, times):  # create average result

    name_list = sorted(os.listdir(workdir + "/resultset/test/round1/result" + str(index) + "/"))

    os.system("rm -rf " + workdir + "/result_set/test/result" + str(index) + "/")
    os.makedirs(workdir + "/result_set/test/result" + str(index) + "/")

    for name in name_list:
        create_single_average_result(workdir, index, times, name, "test")



if __name__=="__main__":

    create_average_result(sys.argv[1])





