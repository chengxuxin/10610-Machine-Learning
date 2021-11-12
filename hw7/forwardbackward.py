import numpy as np
import sys

from numpy.lib.scimath import log

def logsumexp_mat(mat):
        ms = np.max(mat, axis=0)
        return ms + np.log(np.sum(np.exp(mat-ms[None, :]), axis=0))  # sum is over columns

def logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v-m)))


class ForwardBackward():
    def __init__(self) -> None:
        self.init = np.genfromtxt(hmm_init)
        self.trans = np.genfromtxt(hmm_trans)
        self.emit = np.genfromtxt(hmm_emit)
        self.words = np.genfromtxt(index_to_word, dtype=str)
        self.tags = np.genfromtxt(index_to_tag, dtype=str)
        self.word_dict = dict(zip(self.words, np.arange(0, self.words.shape[0])))
        self.tag_dict = dict(zip(self.tags, np.arange(0, self.tags.shape[0])))
        self.num_words = self.words.shape[0]
        self.num_tags = self.tags.shape[0]

        self.loginit = np.log(self.init)
        self.logtrans = np.log(self.trans)
        self.logemit = np.log(self.emit)

        self.all_sentece = self.loadData()

    def loadData(self):
        with open(val_input, 'r') as f:
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
        return all_sentence

    def predict(self, ):
        logprobs = []
        preds = []
        for sen in self.all_sentece:
            words = sen[:, 0]
            sen_len = words.shape[0]

            alphas = np.empty((self.num_tags, sen_len))
            alphas[:, 0] = self.loginit + self.logemit[:, words[0]]
            for i in range(1, sen_len, 1):
                alphas[:, i] = self.logemit[:, words[i]] + logsumexp_mat(self.logtrans + alphas[:, i-1][:, None])
            
            betas = np.empty((self.num_tags, sen_len))
            betas[:, -1] = 0  # log1=0
            for i in range(sen_len-2, -1, -1):
                betas[:, i] = logsumexp_mat(self.logemit[:, words[i+1]][:, None] + self.logtrans.T + betas[:, i+1][:, None])
            
            logp_tags = alphas + betas
            # p_tags = np.exp(logp_tags)
            print(np.exp(alphas))
            print(logsumexp(logp_tags[:, -1]))

            tag_pred = np.argmax(logp_tags, axis=0)
            
            preds.append(tag_pred)
            logprobs.append(logsumexp(alphas[:, -1]))
        return preds, logprobs

if __name__ == "__main__":
    # val_input = sys.argv[1]
    # index_to_word = sys.argv[2]
    # index_to_tag = sys.argv[3]
    # hmm_init = sys.argv[4]
    # hmm_emit = sys.argv[5]
    # hmm_trans = sys.argv[6]
    # pred_file = sys.argv[7]
    # metric_file = sys.argv[8]

    parent_name = "handout/toy_data/"
    val_input = parent_name + "validation.txt"
    index_to_word = parent_name + "index_to_word.txt"
    index_to_tag = parent_name + "index_to_tag.txt"
    hmm_init = "OUTPUT_hmm_init.txt"
    hmm_emit = "OUTPUT_hmm_emit.txt"
    hmm_trans = "OUTPUT_hmm_trans.txt"
    pred_file = "OUTPUT_pred.txt"
    metric_file = "OUTPUT_metric.txt"

    fb = ForwardBackward()
    preds, logprobs = fb.predict()

    ref_tags = np.concatenate([sen[:, 1] for sen in fb.all_sentece]).flatten()
    pred_tags = np.concatenate(preds)
    accuracy = np.mean(pred_tags == ref_tags)
    meanlogprob = np.mean(logprobs)
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: {:.10f}\nAccuracy: {:.10f}".format(meanlogprob, accuracy))

    with open(pred_file, "w") as f:
        for i, sen in enumerate(fb.all_sentece):
            for j, one in enumerate(sen):
                f.write(fb.words[one[0]] + "\t" + fb.tags[preds[i][j]] + "\n")
            f.write("\n")
    