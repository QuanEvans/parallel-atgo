import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import esm
import os
from Bio import SeqIO
import random as rd
from tqdm import tqdm
import json
import numpy as np

prefix_folder = "/media/oof/LabData/"
prefix_folder = "/mnt/f/"

def extract(
            fasta_file:str, # path of fasta file to extract features from
            repr_layers:list = [31,32,33], # which layers to extract features from
            model_path:str = None, # path to model
            model_name:str = None, # name of model, if model_path is not provided
            use_gpu:bool = True, # use GPU if available
            truncate:bool = True, # truncate sequences longer than 1024 to match training setup
            include:list = ["mean", "per_tok", "bos", "contacts"], # which representations to return
            batch_size:int = 4096, # maximum batch size
            output_dir:str = None, # output directory for extracted representations, if None, return as dict
            ) -> dict:
    
    if model_path and os.path.exists(model_path):
        model, alphabet = pretrained.load_model_and_alphabet(model_path)
    elif model_name:
        #model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        raise ValueError("model_path or model_name must be provided")
    model.eval()

    if torch.cuda.is_available() and use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model = model.cuda()
        print("Transferred model to GPU")
    else:
        print("Using CPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():
        if output_dir is None:
            # if no output directory is specified, return the results as a dict
            results_dict = {}
        for batch_idx, (lables, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and use_gpu:
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts="contacts" in include)

            #logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if "contacts" in include:
                contacts = out["contacts"].to(device="cpu")
            

            for i, label in enumerate(lables):

                result = {"label": label}

                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1: len(strs[i]) + 1].clone().numpy().tolist()
                        for layer, t in representations.items()
                    }
                
                if "mean" in include:
                    result["mean_representations"] = {
                    layer: t[i, 1: len(strs[i]) + 1].mean(0).clone().numpy().tolist()
                        for layer, t in representations.items()
                    }

                if "bos" in include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items().numpy().tolist()
                    }

                if "contacts" in include:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone().numpy().tolist()

                if output_dir:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    out_file = os.path.join(output_dir, f"{label}.pt")
                    torch.save(result, out_file)
                else:
                    results_dict[label] = result
        
        if output_dir:
            return None
        else:
            return results_dict


class ModelData:

    def __init__(self, 
                mode:str, # mode of create data, if "raw", class would split data, else, class would load data using fasta_dict
                data_dir:str, # path to data directory that put BP, MF, and CC files
                label_dict:dict, # path to label file for each aspect; in ATGO, this should be gene_GO_Terms, check read_labels for details
                term_dict:dict, # path to term file for each aspect; in ATGO, this should be term_list
                sep:str = "  ", # separator for label file
                fasta_path:str = None, # path to fasta file, required if mode is "raw"s
                fasta_dict:dict = None, # dict key: data type (train,test), value: path to fasta file
                seed = 42, # random seed
                model_name = "esm2_t33_650M_UR50D", # model name for esm
                model_path = None, # path to model directory
                esm_dir = None, # path to esm directory
                ):
        self.check_mode(mode=mode,fasta_path=fasta_path,fasta_dict=fasta_dict)
        self.init_fasta(mode=mode,fasta_path=fasta_path,fasta_dict=fasta_dict,seed=seed)
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.term_dict = term_dict
        self.sep = sep
        self.model_name = model_name
        self.model_path = model_path
        self.aspects = ["BP","MF","CC"]
        self.fasta_dict = fasta_dict
        self.esm_dir = esm_dir

        # initialize dircetory for files
        self.init_dir()
        # dictionary mapping GO terms to aspect
        self.go_aspect = self.get_aspect_term_from_file()
        # initialize protein labels
        self.protein_labels = self.get_aspect_labels()
        # initialize aspect_go2idx
        self.aspect_go2idx = self.get_aspect_go2idx()

    @property
    def BP(self): # biological process go terms
        return self.go_aspect['BP'].difference(self.excludeGOs)
    
    @property
    def MF(self): # molecular function go terms
        return self.go_aspect['MF'].difference(self.excludeGOs)
    
    @property
    def CC(self): # cellular component go terms
        return self.go_aspect['CC'].difference(self.excludeGOs)
    
    @property
    def excludeGOs(self): # GO terms to exclude
        return set("GO:0005515,GO:0005488,GO:0003674,GO:0008150,GO:0005575".split(","))

    def check_mode(self,mode:str, fasta_dict:dict = None,fasta_path:str = None) -> None:
        assert mode in ["raw","load"], f"Mode must be raw or load, not {mode}"
        if mode == "raw":
            assert fasta_path is not None, "fasta_path must be provided if mode is raw"
        else:
            assert fasta_dict is not None, "fasta_dict must be provided if mode is load"
    
    def get_aspect_terms(self,aspect:str) -> set:
        aspect = aspect.upper()
        assert aspect in self.aspects, f"Aspect must be one of BP, MF, or CC, not {aspect}"
        if aspect == "BP":
            return self.BP
        elif aspect == "MF":
            return self.MF
        elif aspect == "CC":
            return self.CC
        
    def init_fasta(self,mode:str,fasta_path:str,seed:int, fasta_dict:dict) -> None:
        if mode == "raw":
            raise NotImplementedError

        elif mode == "load":
            new_fasta_dict = {}
            for key, path in fasta_dict.items():
                new_fasta_dict[key] = self.read_fasta(path)
            self.seq_dict = new_fasta_dict
        
    def get_aspect_term_from_file(self):
        """
        Get aspect term from file
        """
        go_aspect_term = {}
        for aspect in self.aspects:
            with open(self.term_dict[aspect],"r") as f:
                term_list = f.read().splitlines()
                go_aspect_term[aspect] = set(term_list)
        return go_aspect_term
    
    def reverse_dict(self, d: dict) -> dict:
        """
        Reverse a dictionary

        Args:
            d (dict): dictionary to reverse
        returns:
            rev_d (dict): reversed dictionary
        """
        rev_d = {}
        for k, v in d.items():
            if v not in rev_d:
                rev_d[v] = set()
            rev_d[v].add(k)
        return rev_d
    
    def get_aspect_labels(self):
        """
        Get labels for each aspect

        returns:
            aspect_labels (dict): dictionary mapping aspect to labels
        """
        aspect_labels = {}
        for aspect in self.aspects:
            aspect_labels[aspect] = self.read_labels(self.label_dict[aspect],sep=self.sep)
        return aspect_labels
    
    def read_labels(self, file_path:str, sep:str='\t') -> dict:
        """
        Read labels from a file

        Labels should be in the format:
        protein_name<sep>GO1,GO2,GO3,...

        Args:
            file_path (str): path to file
            sep (str): separator between protein name and GO terms
        returns:
            labels (dict): dictionary mapping GO terms to protein names
        """
        protein_labels = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                protein_name, go_terms = line.strip().split(sep)
                go_terms = go_terms.split(',')
                protein_labels[protein_name] = set(go_terms)
        return protein_labels
    
    def read_fasta(self, file_path:str) -> list:
        """
        Read fasta file using biopython

        Args:
            file_path (str): path to fasta file
        returns:
            list of tuples (protein_name, protein_sequence)
        """
        seq_dict = SeqIO.to_dict(SeqIO.parse(file_path, 'fasta'))
        return [(seq.id, str(seq.seq)) for seq in seq_dict.values()]

    def init_dir(self, del_existing:bool=True) -> None:
        """
        Initialize the directory for the data

        Args:
            del_existing (bool): whether to delete existing directory
        """
        if os.path.exists(self.data_dir):
            if del_existing:
                os.system(f"rm -rf {self.data_dir}")
            else:
                raise ValueError(f"{self.data_dir} already exists")
        os.makedirs(self.data_dir)
        os.makedirs(os.path.join(self.data_dir, 'BP'))
        os.makedirs(os.path.join(self.data_dir, 'MF'))
        os.makedirs(os.path.join(self.data_dir, 'CC'))


    def get_aspect_go2idx(self) -> dict:
        """
        Get dictionary mapping GO term to index

        set self.aspect_go2idx
        key: aspect, value: dictionary mapping GO term to index
        """
        aspect_go2idx = {}
        for aspect in self.aspects:
            term_list = self.go_aspect[aspect].copy()
            term_list = sorted(list(term_list))
            aspect_go2idx[aspect] = {go:idx for idx, go in enumerate(term_list)}
        self.aspect_go2idx = aspect_go2idx
        return aspect_go2idx
    
    def write_fasta(self, filename:str, name_seq:tuple) -> None:
        """
        Write fasta file

        Args:
            filename (str): path to output file
            seq_names (list): list of sequence names
            seqs (list): list of sequences
        """
        with open(filename, 'w') as f:
            for name, seq in name_seq:
                f.write(f">{name}\n{seq}\n")

    def write_feature_label(self) -> None:
        """
        write the train/eval/test protein labels to file
        """
        for aspect in self.aspects:
            out_dir = os.path.join(self.data_dir, aspect)

            # wirte the term list
            print(f"writing {aspect} term list")
            out_term_list = os.path.join(out_dir, 'term_list')
            with open(out_term_list, 'w') as f:
                cur_term_list = self.go_aspect[aspect].copy()
                cur_term_list = sorted(list(cur_term_list))
                for term in cur_term_list:
                    f.write(f"{term}\n")

            # protein names from train/eval/test for gene_GO_terms
            for data_type, proteins_seq in self.seq_dict.items():
                cur_apsect_protein_names = set(self.protein_labels[aspect].keys())
                cur_proteins_seq = [ i for i in proteins_seq if i[0] in cur_apsect_protein_names]
                cur_protein_names = [i[0] for i in cur_proteins_seq]

                out_fasta_name = os.path.join(out_dir, f"{data_type}_seq.fasta")
                self.write_fasta(out_fasta_name, cur_proteins_seq)
                
                # write the labels and one hot
                print(f"writing {aspect} {data_type} labels and one hot")
                out_label = os.path.join(out_dir, f"{data_type}_gene_label")
                out_label_one_hot = os.path.join(out_dir, f"{data_type}_label_one_hot")               
                with open(out_label, 'w') as f_label:
                    with open(out_label_one_hot, 'w') as f_one_hot:
                        out_label = os.path.join(out_dir, f"{data_type}_gene_label")
                        out_label_one_hot = os.path.join(out_dir, f"{data_type}_label_one_hot")
                        for protein_name in cur_protein_names:
                            go_terms = self.protein_labels[aspect][protein_name]
                            # go_terms should within cur_aspect
                            go_terms = set(go_terms).intersection(self.get_aspect_terms(aspect))
                            label_one_hot = [0] * len(self.go_aspect[aspect])
                            for go in go_terms:
                                label_one_hot[self.aspect_go2idx[aspect][go]] = 1
                            f_label.write(f"{protein_name}{self.sep}{','.join(go_terms)}\n")
                            f_one_hot.write(f"{' '.join([str(i) for i in label_one_hot])}\n")

                # write the protein list
                print(f"writing {aspect} {data_type} protein list")
                out_protein_list = os.path.join(out_dir, f"{data_type}_gene_list")
                with open(out_protein_list, 'w') as f:
                    for protein_name in cur_protein_names:
                        f.write(f"{protein_name}\n")
                    
                # write the protein features
                print(f"writing {aspect} {data_type} protein features")
                out_protein_feature = os.path.join(out_dir, f"{data_type}_feature")
                with open(out_protein_feature, 'w') as f:
                    layers = [0,35,36]
                    layers_features = []
                    for protein_name in tqdm(cur_protein_names):
                        pt_path = os.path.join(self.esm_dir, f"{protein_name}.pt")
                        layers_feature = self.read_pt(pt_path, layers)
                        layers_features.append(layers_feature)
                    
                    for layers_feature in tqdm(layers_features):
                        f.write(f"{' '.join([str(i) for i in layers_feature])}\n")

                # with open(out_protein_feature, 'w') as f:
                #     layers = [0,35,36]
                #     feature_dict = extract(out_fasta_name, include=['mean'], repr_layers=layers, model_name=self.model_name, model_path=self.model_path)
                #     for protein_name in cur_protein_names:
                #         protein_feature_dict = feature_dict[protein_name]
                #         protein_features = []
                #         for layer in layers:
                #             cur_layer_features = protein_feature_dict['mean_representations'][layer]
                #             protein_features.extend(cur_layer_features)
                #         f.write(f"{' '.join([str(i) for i in protein_features])}\n")
            
            # wirte proein_GO_terms
            print(f"writing {aspect} gene_GO_terms")
            out_protein_GO_terms = os.path.join(out_dir, 'gene_GO_terms')
            with open(out_protein_GO_terms, 'w') as f:
                for name, go_terms in self.protein_labels[aspect].items():
                    f.write(f"{name}{self.sep}{','.join(go_terms)}\n")
    
    def read_pt(self, filename:str, layer_list:list=[0,35,36]) -> tuple:
        """
        Read pt file

        Args:
            filename (str): path to pt file

        Returns:
            tuple: protein names and sequences
        """
        pt_obj = torch.load(filename)
        mean_representations = pt_obj['mean_representations']
        layer_tensors = []
        for layer in layer_list:
            layer_tensors.append(mean_representations[layer].numpy())
        # concat layer tensors
        layer_tensors = np.concatenate(layer_tensors, axis=0)
        return layer_tensors



if __name__=="__main__":
    mode = 'load'
    prefix = f'{prefix_folder}ATGO'
    data_dir = f'{prefix}/re-train/dataset/ours_esm2_3b'
    label_dict = {
        "BP" : f'{prefix}/re-train/dataset/data/BP/gene_GO_Terms',
        "CC" : f'{prefix}/re-train/dataset/data/CC/gene_GO_Terms',
        "MF" : f'{prefix}/re-train/dataset/data/MF/gene_GO_Terms', 
    }
    term_list_dict = {
        "BP" : f'{prefix}/re-train/dataset/data/BP/term_list',
        "CC" : f'{prefix}/re-train/dataset/data/CC/term_list',
        "MF" : f'{prefix}/re-train/dataset/data/MF/term_list',
    }
    #obo_file = f'{prefix}/re-train/dataset/data/go-basic.obo'
    fasta_dict = {
        'train': f'{prefix}/re-train/dataset/data/fasta/train_sequence.fasta',
        'evaluate': f'{prefix}/re-train/dataset/data/fasta/evaluate_sequence.fasta',
        'test': f'{prefix}/re-train/dataset/data/fasta/test_sequence.fasta',
    }
    esm_dir = f'{prefix}/re-train/dataset/esm_feature'
    model_data = ModelData(
        mode=mode,
        data_dir=data_dir,
        term_dict=term_list_dict,
        label_dict=label_dict,
        fasta_dict=fasta_dict,
        esm_dir=esm_dir,
    )
    model_data.write_feature_label()

                

        