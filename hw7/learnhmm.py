import numpy as np
import sys


class HMM():
    def __init__(self):
        self.words = np.genfromtxt(index_to_word, dtype=str)
        self.tags = np.genfromtxt(index_to_tag, dtype=str)
        self.word_dict = dict(zip(self.words, np.arange(0, self.words.shape[0])))
        self.tag_dict = dict(zip(self.tags, np.arange(0, self.tags.shape[0])))
        self.num_words = self.words.shape[0]
        self.num_tags = self.tags.shape[0]

        self.all_sentece = self.loadData()
        self.init, self.trans, self.emit = self.get_matrices()
        # print(self.init, self.trans, self.emit)

    def loadData(self):
        with open(train_input, 'r') as f:
            sentence = []
            all_sentence = []
            for line in f.readlines():
                if line != ("\n"):
                    word_str, tag_str = line.split("\n")[0].split("\t")
                    word_idx = self.word_dict[word_str]
                    tag_idx = self.tag_dict[tag_str]
                    sentence.append([word_idx, tag_idx])
                else:
                    all_sentence.append(np.array(sentence))
                    sentence = []
            all_sentence.append(np.array(sentence))  # last sentence
        # print(all_sentence)
        # print(len(all_sentence))
        return all_sentence

    def get_matrices(self,):
        init = np.zeros(self.num_tags)
        trans = np.zeros((self.num_tags, self.num_tags))
        emit = np.zeros((self.num_tags, self.num_words))
        for sen in self.all_sentece:
            init[sen[0, 1]] += 1
            for i, one in enumerate(sen):
                if i < sen.shape[0]-1:
                    trans[sen[i,1], sen[i+1, 1]] += 1
                emit[sen[i,1], sen[i,0]] += 1
        init += 1
        trans += 1
        emit += 1
        init /= np.sum(init)
        trans /= np.sum(trans, axis=1)[:, None]
        emit /= np.sum(emit, axis=1)[:, None]
        return init, trans, emit

if __name__ == "__main__":
    # train_input = sys.argv[1]
    # index_to_word = sys.argv[2]
    # index_to_tag = sys.argv[3]
    # hmm_init = sys.argv[4]
    # hmm_emit = sys.argv[5]
    # hmm_trans = sys.argv[6]

    parent_name = "handout/toy_data/"
    train_input = parent_name + "train.txt"
    index_to_word = parent_name + "index_to_word.txt"
    index_to_tag = parent_name + "index_to_tag.txt"
    hmm_init = "OUTPUT_hmm_init.txt"
    hmm_emit = "OUTPUT_hmm_emit.txt"
    hmm_trans = "OUTPUT_hmm_trans.txt"
    
    
    hmm = HMM()

    np.savetxt(hmm_init, hmm.init, fmt="%.10f")
    np.savetxt(hmm_trans, hmm.trans, fmt="%.10f")
    np.savetxt(hmm_emit, hmm.emit, fmt="%.10f")

    # compare with reference output
    # ref_init = "handout/en_output/hmminit.txt"
    # ref_trans = "handout/en_output/hmmtrans.txt"
    # ref_emit = "handout/en_output/hmmemit.txt"

    # init = np.genfromtxt(ref_init)
    # trans = np.genfromtxt(ref_trans)
    # emit = np.genfromtxt(ref_emit)

    # print(np.mean(np.abs(init - hmm.init)))
    # print(np.mean(np.abs(trans - hmm.trans)))
    # print(np.mean(np.abs(emit - hmm.emit)))