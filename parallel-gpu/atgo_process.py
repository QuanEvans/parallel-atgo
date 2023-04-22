import os
import sys
from configure import script_dir, python_dir, database_dir, num_threads
import zipfile
import time
from tools.run_blast import *
from Bio import SeqIO
import multiprocessing as mp
from multiprocessing import Pool
from contextlib import closing

num_threads = min(num_threads, mp.cpu_count()-4)
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_seq_dict(fast_file):
    seq_dict = {}
    for record in SeqIO.parse(fast_file, "fasta"):
        seq_dict[record.id] = str(record.seq)
    return seq_dict

def split_sequence(workdir):  # split one sequence file as mutiple sequence files

    seq_file = workdir + "/seq.fasta"
    name_file = workdir + "/name_list"

    f = open(seq_file, "r")
    text = f.read()
    f.close()

    sequence_dict = get_seq_dict(seq_file)

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

def extract_protein_list(fast_file,workdir,database_dir,threads=26, add_self=False, add_seq=False):
    aspects = ['MF', 'BP', 'CC']
    for aspect in aspects:
        database = os.path.join(database_dir, aspect, 'sequence.fasta')
        tsv_out = os.path.join(workdir, f'{aspect}_all_protein_list.tsv')
        run_blast(fast_file, database,threads=threads,add_self=False,add_seq=False,output_file=tsv_out)

def wirte_all_protein_list(workdir, sequence_dict, seq_file, result_dir):
    args_list = []
    aspect = ['MF', 'BP', 'CC']
    for i in range(len(aspect)):
        cur_aspect = aspect[i]
        tsv_out = os.path.join(workdir, f'{cur_aspect}_all_protein_list.tsv')
        df_result = pd.read_csv(tsv_out, sep='\t')
        for name in sequence_dict:
            args_list.append((df_result, name, cur_aspect, result_dir))

    with closing(Pool(num_threads)) as pool:
        pool.map(wirte_protein_list, args_list)
        pool.close()
        pool.join()


def wirte_protein_list(args):
    df, name, aspect, result_dir = args
    sub_dir = result_dir + "/" + name + "/"
    cur_result = df[df['query'] == name]
    cur_result = cur_result[cur_result['query'] != cur_result['target']]
    # sort by bitscore,highest first
    cur_result = cur_result.sort_values(by='bitscore', ascending=False)
    cur_out_name = os.path.join(sub_dir, f'{aspect}_protein_list')
    with open(cur_out_name, 'w') as out:
        for i, row in cur_result.iterrows():
            row_target = row['target']
            row_score = row['bitscore']
            out.write(f'{row_target}\t{row_score}\n')

def zipDir(dirpath,outFullName):
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def all_process(workdir):

    time_0 = time.time()
    split_sequence(workdir)
    fasta_file = workdir + "/seq.fasta"
    extract_protein_list(fasta_file, workdir, database_dir, threads=num_threads)
    sequence_dict = get_seq_dict(fasta_file)
    wirte_all_protein_list(workdir, sequence_dict, fasta_file, workdir + "/SAGP/")

    cmd = "python " + script_dir + "/Pipeline_New_Process.py " + workdir + "/SAGP/"
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

