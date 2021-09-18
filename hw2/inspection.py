import sys, os, csv
import numpy as np

def read(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    array = []
    for row in read_tsv:
        array.append(row)
    nparr = np.array(array)
    return nparr[1:]  # first row is header

def calc_entropy(counts):
    ent = 0
    total = np.sum(counts)
    for count in counts:
        ent -= count/total * np.log2(count/total)
    return ent

def inspection():
    data = read(input)
    data_category = data[:, -1]
    unique_category, counts = np.unique(data_category, return_counts=True)
    n_category = unique_category.shape[0]
    
    ent = calc_entropy(counts)

    major_counts = np.max(counts)
    error = 1 - major_counts / np.sum(counts)

    print(ent, error)
    with open(output, 'w') as f_out:
        f_out.write("entropy: {:f}\nerror: {:f}".format(ent, error))


if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    # input = "./handout/small_train.tsv"
    # output = "./handout/small_inspect.txt"
    inspection()