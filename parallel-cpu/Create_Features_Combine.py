import os
import sys
import numpy as np
from decimal import Decimal

def read_protein_list(protein_list_file):

    f = open(protein_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def create_feature(feature_dir1, feature_dir2, feature_dir3, name_list, feature_file):

    protein_list = read_protein_list(name_list)

    f = open(feature_file, "w")
    for protein in protein_list:

        feature_file = feature_dir1 + "/" +protein
        f1 = open(feature_file, "r")
        text = f1.read()
        f1.close()

        value_list = []
        for data in text.splitlines():
            value_list.append(float(data.strip()))

        line = ""
        for value in value_list:
            line = line + str(Decimal(float(value)).quantize(Decimal("0.00000"))) + " "

        feature_file = feature_dir2 + "/" + protein
        f1 = open(feature_file, "r")
        text = f1.read()
        f1.close()

        value_list = []
        for data in text.splitlines():
            value_list.append(float(data.strip()))

        for value in value_list:
            line = line + str(Decimal(float(value)).quantize(Decimal("0.00000"))) + " "

        feature_file = feature_dir3 + "/" + protein
        f1 = open(feature_file, "r")
        text = f1.read()
        f1.close()

        value_list = []
        for data in text.splitlines():
            value_list.append(float(data.strip()))

        for value in value_list:
            line = line + str(Decimal(float(value)).quantize(Decimal("0.00000"))) + " "

        line = line.strip()
        f.write(line+"\n")
    f.flush()
    f.close()







