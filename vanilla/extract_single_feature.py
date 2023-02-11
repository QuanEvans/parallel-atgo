import numpy as np
import torch
import sys
import os

def extract_features(workdir):

    f = open(workdir + "/name_list", "r")
    text = f.read()
    f.close()
    name_list = text.splitlines()

    for index in range(31, 34):

        dealdir = workdir + "/feature_embeddings/" + str(index) + "/"
        if(os.path.exists(dealdir)==False):
            os.makedirs(dealdir)

        for name in name_list:

            originfile = workdir + "/feature_embeddings/mean/" + name + ".pt"
            data = torch.load(originfile)
            data = data["mean_representations"][index].numpy()
            dealfile = dealdir + "/" + name
            np.savetxt(dealfile, data, fmt="%.6f")


if __name__=="__main__":

    workdir = sys.argv[1]
    extract_features(workdir)
