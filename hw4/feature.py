import numpy as np
import sys, os, csv

def read_reviews1(filename, ref_words, ref_idxs, num_ref_words):
    with open(filename) as f:
        num_lines = sum(1 for line in f)
    all_features = np.zeros((num_lines, num_ref_words), dtype=int)
    labels = np.empty(num_lines, dtype=int)
    sorted_index = np.argsort(ref_words)
    ref_words_sorted = ref_words[sorted_index]
    with open(filename) as f:
        for i, line in enumerate(f):
            label, review = line.split("\t")
            labels[i] = int(label)
            words = review.split(" ")
            for word in words:
                idx = np.searchsorted(ref_words_sorted, word)
                if idx < ref_words_sorted.shape[0]:
                    if ref_words_sorted[idx] == word:
                        all_features[i, ref_idxs[sorted_index[idx]]] = 1
                # idx = np.argwhere(word == ref_words).squeeze()
                # if idx.size != 0:
                #     all_features[i, ref_idxs[idx]] = 1
    return labels, all_features

def read_dict(filename):
    words = []
    idxs =[]
    with open(filename) as f:
        for line in f:
            (word,idx) = line.split(" ")
            words.append(word)
            idxs.append(int(idx))
    return np.array(words), np.array(idxs)

def read_vec_dict(filename):
    data = np.genfromtxt(filename, delimiter="\t", dtype=str, comments=None)
    # with open(filename) as f:
    #     tsv = list(csv.reader(f, delimiter="\t"))
    # data = np.array(tsv)
    words = data[:, 0]
    vecs = data[:, 1:].astype(float)
    # print(data.shape)
    return words, vecs

def read_reviews2(filename, ref_words, ref_vecs, len_vec):
    with open(filename) as f:
        num_lines = sum(1 for line in f)
    all_features = np.empty((num_lines, len_vec), dtype=float)
    labels = np.empty(num_lines, dtype=int)
    sorted_index = np.argsort(ref_words)
    ref_words_sorted = ref_words[sorted_index]
    # print(ref_words_sorted)
    with open(filename) as f:
        for i, line in enumerate(f):
            label, review = line.split("\t")
            labels[i] = int(label)
            words = review.split(" ")
            vecs_this_line = []
            for word in words:
                idx = np.searchsorted(ref_words_sorted, word)
                if idx < ref_words_sorted.shape[0]:
                    if ref_words_sorted[idx] == word:
                        # print(ref_vecs[sorted_index[idx]])
                        vecs_this_line.append(ref_vecs[sorted_index[idx]])
                # idx = np.argwhere(word == ref_words).squeeze()
                # if idx.size != 0:
                #     vecs_this_line.append(ref_vecs[idx])
            
            all_features[i] = np.mean(vecs_this_line, axis=0)
    return labels, all_features

def extract_features1(input, output, dict_file):
    words, idxs = read_dict(dict_file)
    num_words = words.shape[0]
    # print(num_words)
    labels, features = read_reviews1(input, words, idxs, num_words)
    print(labels.shape, features.shape)
    formatted_output = np.hstack((np.expand_dims(labels, axis=1), features))
    np.savetxt(output, formatted_output, fmt="%d", delimiter="\t")

def extract_features2(input, output, dict_file):
    words, vecs = read_vec_dict(dict_file)
    labels, features = read_reviews2(input, words, vecs, vecs.shape[1])
    print(words.shape, vecs.shape)
    print(labels.shape, features.shape)
    formatted_output = np.hstack((np.expand_dims(labels, axis=1), features))
    np.savetxt(output, formatted_output, fmt="%.6f", delimiter="\t")

if __name__ == "__main__":
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_val_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    feature_dict_in = sys.argv[9]

    # train_in = "./handout/smalldata/train_data.tsv"
    # val_in = "./handout/smalldata/valid_data.tsv"
    # test_in = "./handout/smalldata/test_data.tsv"
    # dict_in = "./handout/dict.txt"
    # formatted_train_out = "formatted_train_out.tsv"
    # formatted_val_out = "formatted_val_out.tsv"
    # formatted_test_out = "formatted_test_out.tsv"
    # feature_flag = 2
    # feature_dict_in = "./handout/word2vec.txt"

    if feature_flag == 1:
        extract_features1(train_in, formatted_train_out, dict_in)
        extract_features1(val_in, formatted_val_out, dict_in)
        extract_features1(test_in, formatted_test_out, dict_in)
    else:
        assert feature_flag == 2
        extract_features2(train_in, formatted_train_out, feature_dict_in)
        extract_features2(val_in, formatted_val_out, feature_dict_in)
        extract_features2(test_in, formatted_test_out, feature_dict_in)