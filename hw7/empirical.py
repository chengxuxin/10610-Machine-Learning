import numpy as np
import matplotlib.pyplot as plt

def logsumexp_mat(mat):
        ms = np.max(mat, axis=0)
        return ms + np.log(np.sum(np.exp(mat-ms[None, :]), axis=0))  # sum is over columns

def logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v-m)))

class HMM():
    def __init__(self, index_to_word, index_to_tag):
        self.words = np.genfromtxt(index_to_word, dtype=str)
        self.tags = np.genfromtxt(index_to_tag, dtype=str)
        self.word_dict = dict(zip(self.words, np.arange(0, self.words.shape[0])))
        self.tag_dict = dict(zip(self.tags, np.arange(0, self.tags.shape[0])))
        self.num_words = self.words.shape[0]
        self.num_tags = self.tags.shape[0]

    def update(self, length, train_input):
        self.all_sentece = self.loadData(train_input, length, )
        self.init, self.trans, self.emit = self.get_matrices()
        self.loginit = np.log(self.init)
        self.logtrans = np.log(self.trans)
        self.logemit = np.log(self.emit)
    
    def loadData(self, name, length=None, ):
        with open(name, 'r') as f:
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
        if length is None:
            return all_sentence
        return all_sentence[:length]

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
    
    def predict(self, name):
        data = self.loadData(name)
        logprobs = []
        preds = []
        for sen in data:
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

            tag_pred = np.argmax(logp_tags, axis=0)
            
            preds.append(tag_pred)
            logprobs.append(logsumexp(alphas[:, -1]))
        return preds, logprobs

if __name__ == "__main__":
    parent_name = "handout/en_data/"
    train_input = parent_name + "train.txt"
    val_input = parent_name + "validation.txt"
    index_to_word = parent_name + "index_to_word.txt"
    index_to_tag = parent_name + "index_to_tag.txt"

    seq_lens = [10, 100, 1000, 10000]
    
    hmm = HMM()
    log_train = []
    log_val = []
    for seq_len in seq_lens:
        hmm.update(seq_len, train_input)
        print("sequence length", seq_len)
        preds, logprobs = hmm.predict(name=val_input)
        meanlogprob = np.mean(logprobs)
        log_val.append(log_val)
        print(meanlogprob)
        preds, logprobs = hmm.predict(name=train_input)
        meanlogprob = np.mean(logprobs)
        log_train.append(meanlogprob)
        print(meanlogprob)
    
    plt.plot(seq_lens, log_train)
    plt.plot(seq_lens, log_val)
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Log-Likelyhood")
    plt.legend(["Train", "Validation"])
    plt.show()
    