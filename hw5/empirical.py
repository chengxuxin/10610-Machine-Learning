import numpy as np
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x)
    s = np.sum(exp, axis=0)
    return exp / s

def celoss(y, y_hat):
    return - np.dot(y, np.log(y_hat))

def celoss_batch(y, y_hat):
    # print(np.multiply(y, np.log(y_hat)).shape)
    return - np.sum(np.multiply(y, np.log(y_hat)), axis=0)

def load(filename):
    data = np.genfromtxt(filename, delimiter=",")
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return features, labels

def label2onehot(labels, num_sample, num_out):
    one_hot = np.zeros((num_sample, num_out))
    row_idx = np.arange(num_sample)
    one_hot[row_idx, labels] = 1
    return one_hot

def onehot2label(y):
    return np.argmax(y, axis=1)

class NN():
    def __init__(self, features, labels, val_features, val_labels, hidden, lr, num_epoch, init_flag) -> None:
        self.num_hidden = hidden
        self.num_epoch = int(num_epoch)
        self.num_sample = labels.shape[0]
        self.num_output = 4
        self.input_dim = features.shape[1]
        self.lr = lr
        self.inputs = np.hstack((np.ones((self.num_sample, 1)), features))
        # one_hot = np.zeros((self.num_sample, 4))
        # row_idx = np.arange(self.num_sample)
        # one_hot[row_idx, labels] = 1
        self.ys = label2onehot(labels, self.num_sample, self.num_output)
        self.val_inputs = np.hstack((np.ones((val_labels.shape[0], 1)), val_features))
        self.val_ys = label2onehot(val_labels, val_labels.shape[0], self.num_output)

        if init_flag == 1:  # Random
            self.alpha = np.random.uniform(-0.1, 0.1, (hidden, self.input_dim+1))
            self.beta = np.random.uniform(-0.1, 0.1, (self.num_output, hidden+1))
            self.alpha[:, 0] = 0
            self.beta[:, 0] = 0
        else:
            assert init_flag == 2
            self.alpha = np.zeros((hidden, self.input_dim+1))
            self.beta = np.zeros((self.num_output, hidden+1))
        self.s_alpha = np.zeros(self.alpha.shape)
        self.s_beta = np.zeros(self.beta.shape)

    def forward(self, input, y):  # assert only one sample
        a = np.dot(self.alpha, input)
        z = sigmoid(a)
        z = np.hstack((1, z))
        b = np.dot(self.beta, z)
        y_hat = softmax(b)
        loss = celoss(y, y_hat)
        return loss, y_hat, z

    def backward(self, input, y, y_hat, z):
        dl_db = y_hat - y
        dl_dbeta = dl_db[:, None] @ z[None, :]

        dl_dz = dl_db @ self.beta[:, 1:]  # no z0=1
        dl_da = np.multiply(dl_dz, np.multiply(z[1:], 1-z[1:]))
        dl_dalpha = dl_da[:, None] @ input[None, :]
        return dl_dalpha, dl_dbeta

    def adagrad(self, galpha, gbeta):
        self.s_alpha += galpha**2
        self.s_beta += gbeta**2
        return self.lr / np.sqrt(self.s_alpha+1e-5), self.lr / np.sqrt(self.s_beta+1e-5)

    def train(self, ):
        losses = []
        for i in range(self.num_epoch):
            for input, y in zip(self.inputs, self.ys):
                loss, y_hat, z = self.forward(input, y)
                galpha, gbeta = self.backward(input, y, y_hat, z)
                lr_alpha, lr_beta = self.adagrad(galpha, gbeta)
                self.alpha -= np.multiply(lr_alpha, galpha)
                self.beta -= np.multiply(lr_beta, gbeta)
                # print("iter", i)
                # print(self.alpha)
                # print(self.beta)
                if i == 3:
                    break
            loss_train, y_hat_train = self.pred(self.inputs, self.ys)
            # print("epoch={:d} crossentropy(train):{:.6f}".format(i, np.mean(loss_train)))
            loss_val, y_hat_val = self.pred(self.val_inputs, self.val_ys)
            # print("epoch={:d} crossentropy(validation):{:.6f}".format(i, np.mean(loss_val)))

            # print("error(validation):{}".format(self.error_rate(self.val_ys, y_hat)))
            losses.append((loss_train, loss_val))
        return losses, onehot2label(y_hat_train), onehot2label(y_hat_val)

    def pred(self, inputs, y):  # dealing with batch samples
        a = self.alpha @ inputs.T  # dim=hidden*num_samples
        z = sigmoid(a)
        z = np.vstack((np.ones(z.shape[1]), z))
        b = np.dot(self.beta, z) # dim=n_y*num_samples
        y_hat = softmax(b)
        loss = celoss_batch(y.T, y_hat)
        return loss, y_hat.T

    # def error_rate(self, y, y_hat):
    #     error_rate = self.num_output*np.mean(np.abs(y-y_hat) / 2.0) 
    #     return error_rate

if __name__ == "__main__":
    # train_input = sys.argv[1]
    # validation_input = sys.argv[2]
    # train_out = sys.argv[3]
    # validation_out = sys.argv[4]
    # metrics_out = sys.argv[5]
    # num_epoch = int(sys.argv[6])
    # hidden_units = int(sys.argv[7])
    # init_flag = int(sys.argv[8])
    # learning_rate = float(sys.argv[9])

    train_input = "./data/small_train.csv"
    validation_input = "./data/small_val.csv"
    train_out = "./TRAINOUT.labels"
    validation_out = "VALIDOUT.labels"
    metrics_out = "METRICS.txt"
    num_epoch = 100
    hidden_units = 50
    init_flag = 1
    learning_rate = 0.01

    xs, ys = load(train_input)
    xs_val, ys_val = load(validation_input)

    model = NN(xs, ys, xs_val, ys_val, hidden_units, learning_rate, num_epoch, init_flag)
    losses, labels_train, labels_val = model.train()

    with open(metrics_out, "w") as f:
        i=1
        for loss_train, loss_val in losses:
            f.write("epoch={:d} crossentropy(train): {:f}\n".format(i, np.mean(loss_train)))
            f.write("epoch={:d} crossentropy(validation): {:f}\n".format(i, np.mean(loss_val)))
            i+=1
        f.write("error(train): {}\n".format(np.mean(labels_train!=ys)))
        f.write("error(validation): {}".format(np.mean(labels_val!=ys_val)))

    np.savetxt(train_out, labels_train[:, None], fmt='%d')
    np.savetxt(validation_out, labels_val[:, None], fmt='%d')
    

    
