python_dir = "/home/oof/packages/miniconda3/envs/atgo/bin/python"

#python_dir = "/home/evan/anaconda3/envs/atgo/bin/python"
server_dir = "/mnt/f/ATGO/New_Benchmark_history/"

data_dir = server_dir + "/model/"

script_dir = server_dir + "/program/model/"

esm_model_name = "esm1b_t33_650M_UR50S"

obo_url = data_dir + "/go-basic.obo"

database_dir = data_dir + "/Database/"
go_term_dir =  data_dir + "/UniprotGOA/"
blast_file = data_dir + "/blast-2.12.0/bin/blastp"



go_link = "http://amigo.geneontology.org/amigo/term/"
javascript_list = ["/jmol/JSmol.min.js", "/jmol/Jmol2.js", "/3Dmol/3Dmol-min.js"]
dot_dir = server_dir + "/graphviz/bin/"

num_threads = 16 # number of threads to use
num_round = 10 # number of rounds to run