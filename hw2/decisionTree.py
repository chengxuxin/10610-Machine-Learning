import sys, os, csv, copy
import numpy as np
from collections import Counter


def read(filename):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    array = []
    for row in read_tsv:
        array.append(row)
    nparr = np.array(array)
    return nparr  # first row is header

def calc_entropy(counts):
    ent = 0
    total = np.sum(counts)
    for count in counts:
        ent -= count/total * np.log2(count/total)
    return ent

def calc_mutual_info(data, attribute_col):
    # "attribute_col" is the index of the selected attribute
    unique_cat, cat_counts = np.unique(data[:, -1], return_counts=True)
    ent = calc_entropy(cat_counts)  # calculate total entropy
    data_this_attr = data[:, [attribute_col, -1]]
    attrs = data_this_attr[:, 0]
    categories = data_this_attr[:, 1]
    unique_attr, attr_counts = np.unique(attrs, return_counts=True)  # split according to this attribute
    assert unique_attr.shape[0] <= 2 # assert binary classification
    if len(unique_attr) == 1:
        return 0  # H(A) = H(A|X)
    
    cat0 = categories[np.argwhere(attrs == unique_attr[0])]  # slice data into 2 parts
    cat1 = categories[np.argwhere(attrs == unique_attr[1])]

    ent_cond0 = calc_entropy(np.unique(cat0, return_counts=True)[1])
    ent_cond1 = calc_entropy(np.unique(cat1, return_counts=True)[1])

    ent_cond_this_attr = attr_counts[0]/np.sum(attr_counts) * ent_cond0 + attr_counts[1]/np.sum(attr_counts) * ent_cond1

    return ent - ent_cond_this_attr

def majority_vote(data):
    c = Counter(data[:, -1])
    ordered_cat = c.most_common()  # assume binary or just one
    if len(ordered_cat) == 1:  # pure
        return ordered_cat[0][0]
    else:
        cat0, cat1 = ordered_cat
        if cat0[1] == cat1[1]:  # tie
            if cat0[0] > cat1[0]:
                return cat0[0]
            else:
                return cat1[0]
        else:
            return cat0[0]


class Node():
    def __init__(self, data, depth, split_attr_idx, remaining_attrs, child0=None, child1=None):
        self.curr_depth = depth
        self.data = data
        self.split_attr_idx = split_attr_idx
        self.label = None
        self.leaf = False
        self.remaining_attrs = remaining_attrs
        self.split_attr0 = None
        self.split_attr1 = None
        self.child0 = child0
        self.child1 = child1


class DecisionTree():
    def __init__(self, data, max_depth) -> None:  # data include 1 row of header
        self.max_depth = max_depth
        self.num_attrs = data.shape[1] - 1  # -1 for output category
        self.header = data[0]
        self.cat_names = np.unique(data[1:, -1])
        self.root = Node(data[1:], 0, None, np.arange(self.num_attrs).tolist())  # data in node does not contain header

    def train(self, node: Node):
        is_leaf = self.check_leaf(node)
        if is_leaf:
            node.label = majority_vote(node.data)
            node.leaf = True
            return
        else:
            mutual_info = [calc_mutual_info(node.data, i) for i in node.remaining_attrs]
            split_attr_idx = node.remaining_attrs[np.argmax(mutual_info)]  # include breaking ties with 1st attribute
            selected_attrs = node.data[:, split_attr_idx]
            unique_attr, attr_counts = np.unique(selected_attrs, return_counts=True)
            if len(unique_attr) == 1:# and attr_counts[0] ==1:
                node.label = majority_vote(node.data)
                node.leaf = True
                return
            data0 = np.copy(node.data[np.argwhere(selected_attrs == unique_attr[0]).squeeze(axis=1)])
            data1 = np.copy(node.data[np.argwhere(selected_attrs == unique_attr[1]).squeeze(axis=1)])
            # if split, update current node info
            node.remaining_attrs.remove(split_attr_idx)
            node.split_attr_idx = split_attr_idx 
            node.split_attr0 = unique_attr[0]
            node.split_attr1 = unique_attr[1]
            node.curr_depth += 1
            # create children
            node.child0 = Node(data0, node.curr_depth, None, copy.copy(node.remaining_attrs))  # must use a copy!
            node.child1 = Node(data1, node.curr_depth, None, copy.copy(node.remaining_attrs))
            # train children
            self.print_node(node, data0, unique_attr[0])
            self.train(node.child0)
            self.print_node(node, data1, unique_attr[1])
            self.train(node.child1)
    
    def predict(self, attrs):
        return self.walk_tree(self.root, attrs)  # "attrs" is one datapoint without label
        
    def walk_tree(self, node, attrs):
        if node.leaf == True:
            return node.label
        else:
            attr = attrs[node.split_attr_idx]
            if attr == node.split_attr0:
                node = node.child0
            else:
                assert attr == node.split_attr1
                node = node.child1
            return self.walk_tree(node, attrs)

    def print_node(self, node, data_child, unique_attr):
        unique_cat, cat_counts = np.unique(data_child[:, -1], return_counts=True)
        cat0_idx = np.argwhere(self.cat_names == unique_cat[0])
        cat0 = unique_cat[0]
        if cat0_idx == 0:
            cat1 = self.cat_names[1]
        else:
            cat1 = self.cat_names[0]
        print(node.curr_depth*'|'+"{} = {}: [{}: {} / {}: {}]".format(self.header[node.split_attr_idx], unique_attr, cat0, cat_counts[0], cat1, data_child.shape[0] - cat_counts[0]))

    def check_leaf(self, node):
        unique_names, counts = np.unique(node.data[:, -1], return_counts=True)
        if counts.shape[0] == 1 or \
           node.data.shape[0] == 0 or \
           len(node.remaining_attrs) == 0 or \
           node.curr_depth >= self.max_depth: # perfectly classify; no data; used all attributes; reached max depth
            return True
        else:
            return False


def write(labels, filename):
    with open(filename, 'w') as f_out:
        for line in labels:
            f_out.write(line)
            f_out.write('\n')

def save(tree, data, filename, if_write=True):
    labels = []
    true_labels = data[1:, -1]
    for datapt in data[1:, :-1]:
        label = tree.predict(datapt)
        labels.append(label)
    if if_write:
        write(labels, filename) 
    error = np.mean(labels != true_labels)
    return error

if __name__ == "__main__":
    # binary classifier, all attributes and class only have 2 values.
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # train_in = "./handout/mushroom_train.tsv"
    # test_in = "./handout/mushroom_test.tsv"
    # max_depth = 2
    # train_out = "./output/train_out.labels"
    # test_out = "./output/test_out.labels"
    # metrics_out = "./output/metrics_out.txt"
    
    train_data = read(train_in)
    test_data = read(test_in)
    unique_cat, cat_counts = np.unique(train_data[1:, -1], return_counts=True)
    print("[{}: {} / {}: {}]".format(unique_cat[0], cat_counts[0], unique_cat[1], cat_counts[1]))
    
    tree = DecisionTree(train_data, max_depth)
    tree.train(tree.root)

    error_train = save(tree, train_data, train_out)
    error_test = save(tree, test_data, test_out)
    print(error_train, error_test)
    with open(metrics_out, 'w') as f:
        f.write("error(train): {}\nerror(test): {}".format(error_train, error_test))



    
    
