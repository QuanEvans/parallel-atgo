import argparse
import sys
import os
from tools.atgoTool import *



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('fasta_file', type=str, help='fasta file')
    # parser.add_argument('--data_dir', type=str, default=None, help='data directory')
    # parser.add_argument('--batch_size', type=int, default=1000)
    # parser.add_argument('--out', type=str, default='atgo_result.csv', help='output file name')
    # parser.add_argument('--script_dir', type=str, default=None, help='script directory')
    # args = parser.parse_args()
    # fasta_file = args.fasta_file
    # data_dir = args.data_dir
    # batch_size = args.batch_size
    # script_dir = args.script_dir
    # out = args.out
    prefix = "/media/oof/LabData"
    prefix = "/mnt/f"

    fasta_file = f"{prefix}/ATGO/re-train/_new_eval/test_sequence.fasta"
    fasta_file = '/home/oof/LabData/ATGO/CAFA5_target/testsuperset.fasta'
    data_dir = '/home/oof/LabData/ATGO/CAFA5_target'
    batch_size =  31714
    batch_size =  5000
    script_dir = f"{prefix}/ATGO/parallel-atgo/parallel-gpu/"
    #out = f"{prefix}/ATGO/KDS1/KDS1_transcripts/atgo_result.csv"
    out = "/home/oof/LabData/ATGO/CAFA5_target/CAFA5_target_predict.csv"

    atgo = atgoTool(batch_size=batch_size, data_dir=data_dir, fasta_file=fasta_file, script_path=script_dir)
    atgo.create_input()
    atgo.run_atgo()
    atgo.result2csv(csv_file=out, threshold=0.01)


