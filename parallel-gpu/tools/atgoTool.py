import os
import math
from Bio import SeqIO
import time



class atgoTool:

    def __init__(self, batch_size: int = 1000, # number of sequences to process at a single run
                    data_dir = None, # directory to store the input and output files
                    fasta_file = None, # fasta file to process
                    script_path = None, # path to the ATGO program
                    num_threads = 16, # number of threads
                    num_rounds = 10, # number of rounds for ATGO
                    threshold = 0.01 # treshold for ATGO
                    ):

        self.batch_size = batch_size
        
        if script_path is None:
            self.script_path = self.get_script_path()
        else:
            self.script_path = script_path

        if fasta_file is None:
            raise ValueError("fasta_file must be specified")

        self.script_name = 'atgo_process.py'
        self.data_dir = data_dir

        # if data_dir does not exist, create
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)


        self.fasta_file = fasta_file
        self.num_of_seq = len(self.fasta_read(self.fasta_file))
        self.batch_dir_list  = []
        self.num_threads = num_threads
        self.num_rounds = num_rounds

    @property
    def num_batches(self):
        seqs = self.fasta_read(self.fasta_file)
        return math.ceil(len(seqs)/self.batch_size)

    def fasta_read(self, file):
        seq_dict = SeqIO.to_dict(SeqIO.parse(file, 'fasta'))
        return [(seq.id, str(seq.seq)) for seq in seq_dict.values()]
    
    def get_script_path(self):
        # script path is the parent directory of this file
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
        return parent_dir

    def create_batch(self, seqs=None, batch_size=None):
        if seqs is None:
            seqs = self.fasta_read(self.fasta_file)
        if batch_size is None:
            batch_size = self.batch_size
        batch = []
        for i in range(0, len(seqs), batch_size):
            batch.append(seqs[i:i+batch_size])
        return batch
    
    def create_input(self, batch=None):
        if batch is None:
            batch = self.create_batch()
        for i, seqs in enumerate(batch):
            # make a directory for each batch
            batch_dir = os.path.join(self.data_dir, f'batch_{i*self.batch_size}_{i*self.batch_size+len(seqs)}')
            os.makedirs(batch_dir, exist_ok=True)
            # write the fasta file
            fasta_file = os.path.join(batch_dir, 'seq.fasta')
            with open(fasta_file, 'w') as f:
                for seq in seqs:
                    seq_id, seq = seq
                    if '|' in seq_id:
                        seq_id = seq_id.split('|')[1]
                    f.write(f'>{seq_id}\n{seq}\n')

    def get_batch_dir(self, i_batch=None, batch=None):
        batch_dir_list = []
        if self.batch_dir_list:
            return self.batch_dir_list[i_batch]
        if i_batch is None:
            i_batch = 0
        if i_batch >= self.num_batches:
            print(f'Batch {i_batch} does not exist')
            return None
        if batch is None:
            batch = self.create_batch()
        for i, seqs in enumerate(batch):
            batch_dir = os.path.join(self.data_dir, f'batch_{i*self.batch_size}_{i*self.batch_size+len(seqs)}')
            batch_dir_list.append(batch_dir)
        self.batch_dir_list = batch_dir_list
        return batch_dir_list[i_batch]
        
        
    def get_cmd(self, i_batch=None,log_file='log.txt'):
        if i_batch is None:
            i_batch = 0
        if i_batch >= self.num_batches:
            print(f'Batch {i_batch} does not exist')
            return None
        batch_dir = self.get_batch_dir(i_batch)
        cmd = f'python {self.script_path}/{self.script_name} {batch_dir}'
        if log_file is not None:
            log_file = batch_dir + '/' + log_file
            cmd += f' > {log_file}'
        return cmd, batch_dir
    
    def run_atgo(self, i_batch=None, log_file='log.txt', remove_after_run=False, sleep_time=10):
        if i_batch is None:
            i_batch = 0
        for i in range(i_batch, self.num_batches):
            cmd,batch_dir = self.get_cmd(i, log_file)
            print(f'Running batch {i}...\n{cmd}')
            os.system(cmd)
            if remove_after_run:
                rm_cmd = f'rm -rf {batch_dir}/ATGO/'
                os.system(rm_cmd)
            time.sleep(sleep_time)

    def get_result(self, i_batch=None, threshold=0.5):
        result_list = []# directory of ATGO_Plus
        if i_batch is not None:
            num_batches = i_batch
        else:
            num_batches = self.num_batches

        for i in range(self.num_batches):
            batch_dir = self.get_batch_dir(i)
            atgo_Reader = ATGO_Reader(batch_dir, threshold=threshold)
            all_protein = atgo_Reader.read_all_protein()
            if all_protein:
                result_list += all_protein
        return result_list
    
    def get_result_by_batch(self, batch_list, threshold=0.5):
        result_list = []# directory of ATGO_Plus

        for i in batch_list:
            batch_dir = self.get_batch_dir(i)
            atgo_Reader = ATGO_Reader(batch_dir, threshold=threshold)
            all_protein = atgo_Reader.read_all_protein()
            if all_protein:
                for j in all_protein:
                    result_list.append(j)
        return result_list

    
    def result2csv(self, csv_file=None,result=None, i_batch=None, threshold=0.5):
        if csv_file is None:
            csv_file = 'result.csv'

        if i_batch is not None:
            num_batches = i_batch
        else:
            num_batches = self.num_batches

        if result is None:
            result = self.get_result(i_batch = num_batches, threshold=threshold)

        with open(csv_file,'w') as f:
            f.write('gene_name,prediction_type,term_domain,term_id,confidence_score' + '\n')
            for gene_name, prediction_type, term_domain, term_id, confidence_score in result:
                f.write(f'{gene_name},{prediction_type},{term_domain},{term_id},{confidence_score}' + '\n')
    

### ATGO_Reader regions ###            
            
class ATGO_Protein:

    def __init__(self, 
                 name, # protein name
                 workdir, # the ATGO_PLUS directory
                 pred_type, # 'ATGO' or 'ATGO_PLUS' or 'SAGP'
                 threshold=0.5 # threshold for prediction
                    ):
        
        self.__name = name
        self.__wd = workdir
        self.threshold = threshold
        self.pred_type = pred_type
        self.__seqFile = 'seq.txt' # the file name of the protein sequence
        self.__bp_list = []
        self.__cc_list = []
        self.__mf_list = []
    
    @property
    def name(self):
        return self.__name
    
    @property
    def wd(self):
        return self.__wd
    

    @property
    def seq(self):
        if self.__seqFile not in self.__cached:
            with open(os.path.join(self.wd, self.name, self.__seqFile), 'r') as f:
                seq = f.read().split('\n')[1].strip()
            self.__cached[self.__seqFile] = seq
            return seq
        else:
            return self.__cached[self.__seqFile]
    
    @property
    def bp(self):
        if len(self.__bp_list) == 0:
            self.__bp_list = self.read_result(self.get_file2read('BP', self.pred_type))
        return self.__bp_list
    
    @property
    def cc(self):
        if len(self.__cc_list) == 0:
            self.__cc_list = self.read_result(self.get_file2read('CC', self.pred_type))
        return self.__cc_list
    
    @property
    def mf(self):
        if len(self.__mf_list) == 0:
            self.__mf_list = self.read_result(self.get_file2read('MF', self.pred_type))
        return self.__mf_list
    
    @property
    def all(self):
        return self.bp + self.cc + self.mf


    def get_aspect(self, aspect):

        aspect = aspect.upper()

        if aspect == 'BP':
            return self.bp
        elif aspect == 'CC':
            return self.cc
        elif aspect == 'MF':
            return self.mf
        else:
            raise ValueError('aspect should be bp, cc or mf')
    

    def get_file2read(self, go_type:str, pred_type:int = 0):
        if pred_type == 0:
            pred_type = 'ATGO_PLUS'
        elif pred_type == 1:
            pred_type = 'ATGO'
        elif pred_type == 2:
            pred_type = 'SAGP'
        else:
            raise ValueError('pred_type should be 0, 1 or 2')
        
        go_type = go_type.upper()
        if go_type not in ['BP', 'CC', 'MF']:
            raise ValueError('go_type should be BP, CC or MF')
        
        return pred_type + '_' + go_type

    
    def read_result(self, filename):
        result = []

        try:
            #print(os.path.join(self.wd, self.name, filename))
            with open(os.path.join(self.wd, self.name, filename), 'r') as f:
                for line in f:
                    line = line.strip()
                    go_term, place_holder, prob = line.split(' ')
                    prob = float(prob)
                    if prob >= self.threshold:
                        result.append((go_term, prob)) # return a list of tuple (go_term, prob)
        except FileNotFoundError:
            pass
    
        return result



class ATGO_Reader:

    def __init__(self, 
                workdir, # the work directory of ATGO program
                threshold = 0.5 # threshold for prediction
                ):
        self.__wd = workdir
        self.__result_dir = self.__wd + '/ATGO_PLUS'
        self.__protein_results = []
        self.__threshold = threshold

    
    def read_all_protein(self):

        pred_type_dict = { 0: 'ATGO_PLUS', 1: 'ATGO', 2: 'SAGP'}
        aspect = ['BP', 'CC', 'MF']

        if not os.path.exists(self.__result_dir):
            print('The directory of ATGO_PLUS does not exist')
            return None

        for protein_name in os.listdir(self.__result_dir):
            for pred_type, pred_type_name in pred_type_dict.items():
                try:
                    atgoProtein = ATGO_Protein(protein_name, self.__result_dir, pred_type, self.__threshold)
                except FileNotFoundError:
                    print('The file of {} does not exist'.format(protein_name))
                for go_aspect in aspect:
                    for go_term, prob in atgoProtein.get_aspect(go_aspect):
                        single_term_list = [protein_name,pred_type_name,go_aspect,go_term,prob]
                        self.__protein_results.append(single_term_list)

        return self.__protein_results
