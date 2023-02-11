import os
import sys

from configure import script_dir, python_dir

workdir = sys.argv[1]
type_list = ["MF", "BP", "CC"]
index_dict = dict()
index_dict["MF"] = 630
index_dict["BP"] = 580
index_dict["CC"] = 270

for round in range(1, 11):
    for type in type_list:
        os.system(python_dir + " " + script_dir + "/Load_Triplet_Model.py " + workdir + " " + type + " " + str(round) + " " + str(index_dict[type]))