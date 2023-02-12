import os
import sys
from configure import script_dir, python_dir
import zipfile
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def split_sequence(workdir):  # split one sequence file as mutiple sequence files

    seq_file = workdir + "/seq.fasta"
    name_file = workdir + "/name_list"

    f = open(seq_file, "r")
    text = f.read()
    f.close()

    sequence_dict = dict()
    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name = line[1:]
        else:
            sequence_dict[name] = line

    result_dir = workdir+"/SAGP/"
    os.system("rm -rf "+result_dir)
    os.makedirs(result_dir)

    f1 = open(name_file, "w")

    for name in sequence_dict:
        sub_dir = result_dir+"/"+name+"/"
        os.makedirs(sub_dir)

        f = open(sub_dir+"/seq.fasta", "w")
        f.write(">"+name+"\n"+sequence_dict[name])
        f.flush()
        f.close()

        f1.write(name + "\n")
    f1.flush()
    f1.close()


def zipDir(dirpath,outFullName):
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def all_process(workdir):

    split_sequence(workdir)

    time_0 = time.time()
    cmd = "python2 " + script_dir + "/Pipeline_New_Process.py " + workdir + "/SAGP/"
    os.system(cmd)
    time_1 = time.time()
    print("SAGP time: ", time_1 - time_0)
    locattime = time.asctime(time.localtime(time.time()))
    print(f'Finish SAGP at {locattime}')

    cmd = "python " + script_dir + "/Feature_Extraction.py " + workdir
    os.system(cmd)
    time_2 = time.time()
    print("Feature_Extraction time: ", time_2 - time_1)
    locattime = time.asctime(time.localtime(time.time()))
    print(f'Finish Feature_Extraction at {locattime}')

    cmd = python_dir + " " + script_dir + "/Run_Load_Model.py " + workdir + "/ATGO/"
    os.system(cmd)
    time_3 = time.time()
    print("ATGO time: ", time_3 - time_2)
    locattime = time.asctime(time.localtime(time.time()))
    print(f'Finish ATGO at {locattime}')

    cmd = "python " + script_dir + "/Get_Average_result_from_Network.py " + workdir + "/ATGO/"
    os.system(cmd)
    time_4 = time.time()
    print("Get_Average_result_from_Network time: ", time_4 - time_3)
    locattime = time.asctime(time.localtime(time.time()))

    os.system("rm -rf " + workdir + "/ATGO_PLUS/")

    cmd = "python2 " + script_dir + "/Combine_result.py " + workdir
    os.system(cmd)
    time_5 = time.time()
    print("Combine_result time: ", time_5 - time_4)
    locattime = time.asctime(time.localtime(time.time()))




if __name__=="__main__":

    workdir = sys.argv[1]
    all_process(workdir)

