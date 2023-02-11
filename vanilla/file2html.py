#!/usr/bin/env python
docstring='''file2html datadir
    convert COFACTOR output result to user friendly html output

input files (mandatory):
    seq.fasta (input sequence)
    model1.pdb (input structure)

input files (optional):
    input.pdb (original input structure before residue re-numbering)
    homoflag  (whether to remove homologous template)
    similarpdb_model1.lst (top structure homologs in PDB)
    ECsearchresult_*.dat (EC prediction result)
    lr_GOfreqPPICofactor_{MF,BP,CC} (GO prediction)
    BSITE_model1/Bsites_*.dat (ligand binding site prediction result)
'''
import sys,os
import textwrap
import re
from glob import glob
import tarfile
#from string import Template

from cscore2csv import cscore2csv,detect_graphviz,draw_GO_DAG,color_list
from module import obo2csv

from configure import go_link
from configure import javascript_list
from configure import obo_url, dot_dir

from html_template import index_template
from html_template import go_table_template,go_button_template

cscore_threshold_dict={'MF':0.465,'BP':0.230,'CC':0.415}
#cscore_threshold_dict={'MF':0.1,'BP':0.08,'CC':0.1}

def make_html(jobID,target_name,sequence, seq_fasta, lr_file_dict,lr_txt_dict,lr_graph_dict, title_name):
    viewer_size=450 # size of 3Dmol/JSmol

    go_table_dict=dict()
    namespace_dict={"MF":"Molecular Function","BP":"Biological Process",
        "CC":"Cellular Component"}
    for Aspect in namespace_dict:
        colorbar=''
        for cscore in range(9,-1,-1):
            if cscore==10-len(color_list)+1:
                colorbar='<td style="background-color:%s;">[%.2f,%.1f)</td>'%(
                    color_list[9-cscore], min([cscore/10.,
                    cscore_threshold_dict[Aspect]]),(cscore+1)/10.)+colorbar
                break
            else:
                colorbar='<td style="background-color:%s;">[%.1f,%.1f)</td>'%(
                    color_list[9-cscore],cscore/10.,(cscore+1)/10.)+colorbar
        colorbar=colorbar.replace('1.0)','1.0]')

        go_button_list=[]
        for line in lr_txt_dict[Aspect].splitlines():
            term,aspect_short,cscore,name=line.split('\t')
            go_button_list.append(go_button_template.substitute(dict(
                GO_LINK=go_link,TERM=term,CSCORE=cscore,NAME=name,
            )))

        go_table_dict[Aspect]=go_table_template.substitute(dict(
            NAMESPACE=namespace_dict[Aspect],
            ASPECT=Aspect,

            DAG_GRAPH= "ATGO_PLUS/" + jobID + "/GOsearchresult_final_" + Aspect + ".svg",
            SIZE=viewer_size,
            GO_BUTTONS=''.join(go_button_list),
            PRED_FILE= "ATGO_PLUS/" + jobID + "/GOsearchresult_final_" + Aspect + ".csv",
            COLORBAR=colorbar,
            SRC_DIR = "./",
        ))

    html_txt=index_template.substitute(dict(
        JAVASCRIPT0=javascript_list[0],JAVASCRIPT1=javascript_list[1],
        JOBID=jobID,
        CITATION="Yi-Heng Zhu, Chengxin Zhang, Yan Liu, Dong-Jun Yu, Yang Zhang. ATGO: Improving protein function prediction using attention network-based transformer with triplet network. Submit.",
        HTTP_LINK="https://zhanglab.dcmb.med.umich.edu/ATGO/",
        TARGET_NAME=target_name,
        TITLE_NAME = title_name,
        LEN=len(sequence),
        WRAP_SEQUENCE="<br>\n".join(textwrap.wrap(sequence,60)),
        FASTA_INPUT="./ATGO_PLUS/" + jobID + "/seq.txt",
        #PDB_INPUT=os.path.basename(input_pdb),
        #HOMOFLAG=homoflag,
        #TEMPLATE_PDB_FILE="model1_%s.pdb"%similarpdb_list[0][0],
        #JMOL_ROOT=os.path.dirname(javascript_list[0]),
        SIZE=viewer_size,
        #TEMPLATE_BUTTONS=template_buttons,
        #EC_TABLE=ec_table,
        MF_TABLE=go_table_dict["MF"],
        BP_TABLE=go_table_dict["BP"],
        CC_TABLE=go_table_dict["CC"],
        SRC_DIR = "./",
        #LBS_TABLE=lbs_table,
    ))
    return html_txt

def read_single_sequence_from_fasta(seq_txt="seq.txt"):
    '''read single sequence file "seq_txt". return header and sequence'''
    target_name=''
    sequence=''
    fp=open(seq_txt)
    for line in fp.read().splitlines():
        if line.startswith('>'):
            target_name+=line.lstrip('>').strip().split()[0]
        else:
            sequence+=line.strip()
    fp.close()
    return target_name,sequence



def reformat_GOsearchresult(infile,outfile,obo_dict,infmt="GOfreq",min_cscore=0):
    '''use cscore2csv module to reformat GO prediction result'''
    # read GO prediction
    fp=open(infile,'rU')
    GOfreq_txt=fp.read()
    fp.close()

    # reformat GO prediction
    report_txt,report_dict=cscore2csv(GOfreq_txt,obo_dict,
        infmt=infmt,min_cscore=min_cscore)
    if len(report_txt.splitlines())<10:
        report_txt_tmp,report_dict_tmp=cscore2csv(
            GOfreq_txt,obo_dict,infmt=infmt)
        if len(report_txt_tmp.splitlines())<10:
            report_txt=report_txt_tmp
            report_dict=report_dict_tmp
        else:
            report_txt=''
            report_dict=dict()
            for line in report_txt_tmp.strip().splitlines()[:10]:
                report_txt+=line+'\n'
                GOterm,Aspect,Cscore,name=line.split('\t')
                if not Aspect in report_dict:
                    report_dict[Aspect]=dict()
                report_dict[Aspect][GOterm]=float(Cscore)

    # write GO prediction
    fp=open(outfile,'w')
    fp.write(report_txt)
    fp.close()
    return report_txt,report_dict

def file2html(datadir, title_name):
    '''
    [1] re-organize data files, plot DAG for GO prediction
    [2] make HTML output
    [3] make result tarball
    '''
    #### [1] re-organize data files, plot DAG for GO prediction ####
    ## directory structure ##
    datadir=os.path.abspath(datadir)
    jobID=os.path.basename(datadir)

    ## fasta sequence ##
    seq_fasta=os.path.join(datadir,"seq.fasta")
    if not os.path.isfile(seq_fasta):
        seq_fasta=os.path.join(datadir,"seq.txt")
    target_name,sequence=read_single_sequence_from_fasta(seq_fasta)

    fp=open(obo_url,'rU')
    obo_txt=fp.read()
    fp.close()
    obo_dict=obo2csv.parse_obo_txt(obo_txt)
    obo_dict["F"]["uninformative"].append("GO:0005515")

    ## lr_GOfreqPPICofactor_{MF,BP,CC}: GO prediction
    lr_file_dict=dict()  # key - Aspect, value - final GO prediction file
    lr_txt_dict=dict()   # key - Aspect, value - final GO prediction text
    lr_graph_dict=dict() # key - Aspect, value - final GO prediction graph
    GOsearchresult_file_list=[]
    execpath=dot_dir + detect_graphviz() # location of "dot" program from graphviz
    T="svg" # file type for GO prediction graph
    for Aspect in ["MF","BP","CC"]:
        # parse final consensus prediction
        infile=os.path.join(datadir, "ATGO_PLUS_" + Aspect)
        outfile=os.path.join(datadir,"GOsearchresult_final_%s.csv"%Aspect)
        report_txt,report_dict=reformat_GOsearchresult(
            infile,outfile,obo_dict,min_cscore=cscore_threshold_dict[Aspect])
        lr_file_dict[Aspect]=outfile

        # remove low confidence prediction
        lr_txt_dict[Aspect]=''
        cscore_min=cscore_threshold_dict[Aspect]
        term_count=0.
        color_term_list=[] # a list of confidently predicted terms
        for line in report_txt.splitlines():
            term,aspect_short,cscore,name=line.split('\t')
            term_count+=(not term in obo_dict[Aspect[-1]]["uninformative"]
                )*(float(cscore)>cscore_min)
            if float(cscore)<cscore_min:
                if cscore_min<0 or term_count>=10 or (
                    term_count>=10 and Aspect=="BP") or (
                    term_count>=3 and Aspect=="CC"):
                    break # minimze the number of output GO terms
                else:
                    while (float(cscore)<cscore_min and cscore_min>0):
                        cscore_min-=0.01
                    term_count+=1
            lr_txt_dict[Aspect]+=line+'\n'
            color_term_list.append(term)
        
        # draw GO prediction graph
        lr_graph=os.path.join(datadir,
            "GOsearchresult_final_%s.%s"%(Aspect,T))

        print(execpath)
        gv_txt=draw_GO_DAG(report_dict,lr_graph,obo_dict,T,execpath,
            cscore_min=cscore_min,color_term_list=color_term_list,
            color_threshold=cscore_threshold_dict[Aspect])
        fp=open(Aspect+".dot",'w')
        fp.write(gv_txt)
        fp.close()
        lr_graph_dict[Aspect]=lr_graph


    #### [2] make HTML output ####
    index_html=os.path.join(datadir,"index.html")
    html_txt=make_html(jobID,target_name,sequence,seq_fasta,lr_file_dict,lr_txt_dict,lr_graph_dict, title_name)
    fp=open(index_html,'w')
    fp.write(html_txt)
    fp.close()

    return

if __name__ == '__main__':

    file2html(sys.argv[1], sys.argv[2])



