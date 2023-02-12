# parallel-atgo
Protein language model based protein gene ontology prediction neural networks

ATGO is a deep learning-based algorithm for high accuracy protein Gene Ontology (GO) prediction. Starting from a query sequence, it first extracts three layers of feature embeddings from a pre-trained protein language model (ESM-1b). Next, a fully connected neural network is used to fuse the feature embeddings, which are then fed into a supervised triplet network for GO function prediction. Large-scale benchmark tests demonstrated significant advantage of ATGO on protein function annotations due to the integration of discriminative feature embeddings from attention transformer models. 
The server version of ATGO could be accessed throught https://zhanggroup.org/ATGO/.








This project is based on the following research paper:

Title: Integrating unsupervised language model with triplet neural networks for protein gene ontology prediction
Authors: Yi-Heng Zhu, Chengxin Zhang, Dong-Jun Yu, Yang Zhang
Published: December 22, 2022
DOI: https://doi.org/10.1371/journal.pcbi.1010793

