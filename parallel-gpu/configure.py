# python_dir = "/home/oof/programs/miniconda3/envs/atgo/bin/python"
# server_dir = "/media/oof/LabData/ATGO/New_Benchmark_history/"
# script_dir = '/media/oof/LabData/ATGO/parallel-atgo/parallel-gpu'

python_dir = "/home/oof/packages/miniconda3/envs/atgo/bin/python"
#server_dir = "/mnt/f/ATGO/New_Benchmark_history/"
server_dir = "/home/oof/LabData/ATGO/New_Benchmark_history/"
script_dir = '/mnt/f/ATGO/parallel-atgo/parallel-gpu'

#data_dir = server_dir + "/model/" # this for esm1
data_dir = '/home/oof/LabData/ATGO/re-train/ours/' # this for esm2

esm_model_name = "esm1b_t33_650M_UR50S"

obo_url = data_dir + "/go-basic.obo"

database_dir = data_dir + "/Database/"
go_term_dir =  data_dir + "/UniprotGOA/"
blast_file = data_dir + "/blast-2.12.0/bin/blastp"



go_link = "http://amigo.geneontology.org/amigo/term/"
javascript_list = ["/jmol/JSmol.min.js", "/jmol/Jmol2.js", "/3Dmol/3Dmol-min.js"]
dot_dir = server_dir + "/graphviz/bin/"

num_threads = 24 # number of threads to use
num_round = 1 # number of rounds to run
