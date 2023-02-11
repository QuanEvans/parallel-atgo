import sys
import os

import blast2msa_new as bm
from decimal import Decimal
import Find_Parents as fp
from configure import database_dir, blast_file, go_term_dir

def run_blast(workdir, type):    # run blast

    seq_file = workdir + "/seq.fasta"

    if(os.path.exists(seq_file)==False or os.path.getsize(seq_file)==0):
        print("seq.fasta is not exist")
        return

    xml_file = workdir + "/blast_" + type + ".xml"
    database_file =  database_dir + "/" + type + "/sequence.fasta"

    cmd = blast_file + \
          " -query " + seq_file + \
          " -db " + database_file + \
          " -outfmt 5 -evalue 0.1 " \
          " -out " + xml_file

    os.system(cmd)

def extract_msa(workdir, type): # extract blast

    xml_file = workdir + "/blast_" + type + ".xml"
    seq_file = workdir + "/seq.fasta"
    msa_file = workdir + "/blast_" + type + ".msa"

    if(os.path.exists(xml_file)==False or os.path.getsize(xml_file)==0):
        print("blast.xml is not exist")
        return

    bm.run_extract_msa(seq_file, xml_file, msa_file)

def create_protein_list(workdir, go_dict, type):

    msa_file = workdir + "/blast_" + type + ".msa"

    template_list = []

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            template = line.strip().split("\t")[0][1:]
            score = line.strip().split("\t")[1]
            if(template in go_dict):
                template_list.append([template, score])

    f = open(workdir + "/" + type + "_protein_list", "w")
    for template, score in template_list:
        f.write(template + " " + score + "\n")
    f.flush()
    f.close()


def read_protein_list(protein_list_file):    # read protein templates

    f = open(protein_list_file, "r")
    text = f.read()
    f.close()

    protein_list_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        protein_list_dict[values[0]] = float(values[1])

    return protein_list_dict



def read_go(gofile):  # read GO Terms

    f = open(gofile, "rU")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict



def annotate(workdir, type, obo_dict, go_dict):  # annotate GO term

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

    fp.find_parents_from_file(resultfile, resultfile+"_new", obo_dict)
    fp.sort_result(resultfile)


def process(workdir, obo_dict):   # main process



    type_list = ["MF", "BP", "CC"]

    for type in type_list:

        run_blast(workdir, type)
        extract_msa(workdir, type)

        go_term_file = go_term_dir + "/" + type + "_Term"
        go_dict = read_go(go_term_file)

        create_protein_list(workdir, go_dict, type)
        annotate(workdir, type, obo_dict, go_dict)

if __name__ == '__main__':

    workdir = sys.argv[1]
    obo_dict = fp.get_obo_dict()

    for name in os.listdir(workdir):
        process(workdir + "/" + name + "/", obo_dict)









