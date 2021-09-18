import sys, os
import numpy as np
import csv
from collections import Counter

def tsv2np(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    array = []
    for row in read_tsv:
        array.append(row)
    nparr = np.array(array)
    return nparr[1:]  # first row is header

def train():
    data_array = tsv2np(train_in)
    class_arr = data_array[:, -1]
    selected_attr = data_array[:, split_index]
    attr_class = np.unique(selected_attr)
    assert attr_class.shape == (2,)
    selected_attr = np.where(selected_attr==attr_class[0], 1, 0)
    dataset1_idx = np.where(selected_attr==1)[0]
    dataset0_idx = np.where(selected_attr==0)[0]

    # majority vote
    # classes = np.unique(class_arr)
    classes1 = class_arr[dataset1_idx]
    c = Counter(classes1)
    major_class1 = c.most_common(1)[0][0]

    classes0 = class_arr[dataset0_idx]
    c = Counter(classes0)
    major_class0 = c.most_common(1)[0][0]
    
    return major_class0, major_class1, attr_class

def test(testfile, outfile, major_class0, major_class1, attr_class):
    test_array = tsv2np(testfile)
    selected_attr = test_array[:, split_index]
    selected_attr = np.where(selected_attr==attr_class[0], 1, 0)
    out_class = []
    for attr in selected_attr:
        if attr == 0:
            out_class.append(major_class0)
        else:
            out_class.append(major_class1)
    accuracy_arr = np.array(out_class) == test_array[:, -1]
    error_rate = 1 - np.mean(accuracy_arr)

    # directory = os.path.dirname(outfile)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    with open(outfile, 'w') as f_out:
        for line in out_class:
            f_out.write(line)
            f_out.write('\n')
    return error_rate


def decisionStump():
    major_class0, major_class1, attr_class = train()

    err_train = test(train_in, train_out, major_class0, major_class1, attr_class)
    err_test = test(test_in, test_out, major_class0, major_class1, attr_class)
    
    # directory = os.path.dirname(metrics_out)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    with open(metrics_out, 'w') as f_out:
        f_out.write("error(train): " + str(err_train) + "\nerror(test): " + str(err_test))
    

if __name__ == "__main__":
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # train_in = "./handout/education_train.tsv"
    # test_in = "./handout/education_test.tsv"
    # split_index = 5
    # train_out = "./output/train_out.labels"
    # test_out = "./output/test_out.labels"
    # metrics_out = "./output/metrics_out.txt"

    decisionStump()