import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import logging
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from utils import setup_logger, check_create_dir
from article_corpus import ArticleCorpus
from embedding_interface import IWordEmbedding
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


def plot_loss(loss_history, name="model"):
    """
    plots training loss versus interations

    :param loss_history: list of loss values during training
    :type loss_history: List
    :param name: model name, defaults on "model"
    :type name: str, optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(loss_history)
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    save_path = os.path.join("./data", "{}_loss_history.png".format(name))
    fig.savefig(save_path, dpi=200)
    plt.cla()
    plt.close(fig)
    plt.clf()


class BatchGenerator:
    """
    A class to facilitate training of word embedding models by creating 
    batches of training data 
    """

    def __init__(self, corpus):
        """
        init function for batch generator

        :param corpus: underlying collection of text for training
        :type corpus: ArticleCorpus
        """
        assert isinstance(corpus, ArticleCorpus), "wrong type for corpus"
        self._corpus = corpus

    def generate_batch(self, bs=128, window=5, neg_sampling=False):
        """
        a generator for batches of training data

        :param bs: batch size, defaults to 128
        :type bs: int, optional
        :param window: context window size, defaults to 5
        :type window: int, optional
        :param neg_sampling: flag to indicate negative sampling usage, defaults to False
        :type neg_sampling: bool, optional
        :yield: a batch generator
        :rtype: tuple(List, List, List[List])
        """
        center_ids, outside_ids, negative_ids = [], [], []
        neg_generator = self._corpus.sample_negative_words(num_negative=8)

        while True:
            cur_window = random.randint(1, window)
            center_word, context = self._corpus.get_random_context(cur_window)

            for outside_word in context:
                center_ids.append(self._corpus.word_to_idx[center_word])
                outside_ids.append(self._corpus.word_to_idx[outside_word])
                if neg_sampling:
                    negative_ids.append(next(neg_generator))

                if len(center_ids) == bs:
                    yield center_ids, outside_ids, negative_ids
                    # init next batch
                    center_ids, outside_ids, negative_ids = [], [], []


class Word2VecBase(IWordEmbedding):
    def __init__(self, embedding_dim=16, loss_type="nll"):
        assert loss_type in [
            "nce",
            "nll",
        ], "loss type not supported, available options are 'nce' and 'nll'"

        self.loss_type = loss_type
        self._vocab_size = None
        self._embedding_dim = embedding_dim
        self._mat_center = None
        self._mat_outside = None

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def model_weights(self):
        if isinstance(self._mat_center, np.ndarray):
            w = np.concatenate((self._mat_center, self._mat_outside), axis=0)
            return w
        else:
            return None

    @staticmethod
    def sigmoid(x):
        """
        compute sigmoid function of the input

        :param x: input value
        :type x: float or np.ndarray
        :return: element wise sigmoid function applied on input
        :rtype: float or np.ndarray
        """
        s = 1.0 / (1 + np.exp(-x))
        return s

    @staticmethod
    def softmax(x):
        """
        compute softmax of the input 

        :param x: input logits
        :type x: np.ndarray
        :return: computed probability from logits using softmax
        :rtype: np.ndarray
        """
        x -= np.expand_dims(
            np.max(x, axis=-1), axis=-1
        )  # numerical trick for stability
        x = np.exp(x)
        factor = np.expand_dims(np.sum(x, axis=-1), axis=-1)
        x /= factor
        return x

    def _init_embeddings(self):
        self._mat_center = (
            np.random.rand(self._vocab_size, self._embedding_dim) - 0.5
        )  # / self._embedding_dim
        # self._mat_outside = np.zeros((self._vocab_size, self._embedding_dim))
        self._mat_outside = np.random.rand(self._vocab_size, self._embedding_dim) - 0.5

    def _get_nll_loss_and_grad(self, center_idx, outside_idx):
        # get outside word vector -> (embedding_dim, )
        u_o = self._mat_outside[outside_idx]
        # get center word vector -> (embedding_dim, )
        v_c = self._mat_center[center_idx]
        # get softmax probability -> (1, vocab_size)
        p_i = self.softmax(np.dot(self._mat_outside, v_c.reshape(-1, 1)).T)
        # get negative log likelihood loss (NLL Loss)
        loss = -np.log(p_i[0, outside_idx])  # scalar
        # get gradient wrt center embedding matrix
        grad_center = np.zeros((self._vocab_size, self._embedding_dim))
        grad_center[center_idx, :] = -u_o + np.dot(p_i, self._mat_outside).squeeze()
        # get gradient wrt outside embedding matrix
        p_i[0, outside_idx] -= 1
        grad_outside = np.outer(p_i.squeeze(), v_c)
        # get gradient wrt model weights -> (2*vocab_size, embedding_dim)
        grad_weights = np.concatenate((grad_center, grad_outside), axis=0)
        return loss, grad_weights

    def _get_nce_loss_and_grad(self, center_idx, outside_idx):
        num_negative = 10
        negative_indices = []

        # negative sampling
        while len(negative_indices) < num_negative:
            new_idx = random.randint(0, self._vocab_size - 1)
            if new_idx != outside_idx:
                negative_indices.append(new_idx)
        indices = [outside_idx] + negative_indices

        u_o = self._mat_outside[outside_idx]  # (d, )
        U_k = self._mat_outside[negative_indices]  # (k, d)
        v_c = self._mat_center[center_idx]
        prob_indices = np.hstack(
            [self.sigmoid(np.dot(u_o, v_c)), self.sigmoid(np.dot(U_k, v_c))]
        )

        p = np.zeros(shape=(1, self._vocab_size))
        for i, idx in enumerate(indices):
            p[0, idx] += prob_indices[i]
        p[0, outside_idx] -= 1

        # LOSS
        loss = (
            -np.log(self.sigmoid(np.dot(u_o, v_c)))
            - np.log(self.sigmoid(-np.dot(U_k, v_c))).sum()
        )
        grad_center = np.zeros((self._vocab_size, self._embedding_dim))
        grad_center[center_idx, :] = np.dot(p, self._mat_outside).squeeze()
        grad_outside = np.outer(p.squeeze(), v_c)
        grad_weights = np.concatenate((grad_center, grad_outside), axis=0)
        return loss, grad_weights

    def _train(self, corpus, bs=128, alpha=0.25, max_iter=5000):
        # check folder existence for storing model weights
        check_create_dir("./data")
        max_window_size = 8

        def train_one_batch(cur_alpha):
            loss = 0
            grad = np.zeros(self.model_weights.shape)

            for _ in range(bs):
                window = random.randint(1, max_window_size)
                center_word, context = corpus.get_random_context(window)

                for outside_word in context:
                    center_idx = corpus.word_to_idx[center_word]
                    outside_idx = corpus.word_to_idx[outside_word]
                    if self.loss_type == "nll":
                        this_loss, this_grad = self._get_nll_loss_and_grad(
                            center_idx, outside_idx
                        )
                    else:
                        this_loss, this_grad = self._get_nce_loss_and_grad(
                            center_idx, outside_idx
                        )
                    loss += this_loss / bs
                    grad += this_grad / bs

            # update weights (sgd)
            self._mat_center -= cur_alpha * grad[: self._vocab_size, :]
            self._mat_outside -= cur_alpha * grad[self._vocab_size :, :]
            return loss

        # training
        print_interval = 10
        save_interval = 500
        anneal_interval = 1000

        loss_history = []
        start_time = time()
        cur_alpha = alpha

        for i in range(1, max_iter):
            cur_loss = train_one_batch(cur_alpha)
            loss_history.append(cur_loss)

            if (i % print_interval) == 0:
                duration = (time() - start_time) / 60.0
                logger.info(
                    "Iteration: {} | Loss: {:.2f} | Elapsed Time: {:.1f} mins".format(
                        i, cur_loss, duration
                    )
                )
                # plot losses
                plt.plot(loss_history)
                plt.xlabel("Batch Number")
                plt.ylabel("Loss")
                plt.title("Loss History")
                plt.savefig(
                    "./data/train_history_word2vec_{}.png".format(self.loss_type),
                    dpi=200,
                )
                plt.clf()

            if (i % save_interval) == 0:
                logger.info("saving model weights...")
                file_path = "./data/word2vec_{}.npy".format(self.loss_type)
                np.save(file_path, self.model_weights)

            if (i % anneal_interval) == 0:
                cur_alpha *= 0.5
                logger.info("decreasing the learning rate to {:.4f}".format(cur_alpha))
        return loss_history

    def build_embeddings(self, corpus):
        self._vocab_size = corpus.vocab_size
        self._word_to_idx = corpus.word_to_idx
        self._init_embeddings()
        losses = self._train(corpus)

        plt.plot(losses)
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.savefig(
            "./data/train_history_word2vec_{}.png".format(self.loss_type), dpi=200
        )
        plt.clf()

    def get_embeddings(self, words):
        indices = [self._word_to_idx[word] for word in words]
        word_vectors = self._mat_center[indices, :]
        return word_vectors


class Word2VecTorch(nn.Module, IWordEmbedding):
    """
    Implements word2vec skip-gram model with negative sampling (SGNS) using pytorch
    """

    def __init__(self, corpus, embedding_dim=16):
        """
        initialization of the SGNS model 

        :param corpus: underlying corpus text
        :type corpus: ArticleCorpus
        :param embedding_dim: embedding dimension, defaults to 16
        :type embedding_dim: int, optional
        """
        super(Word2VecTorch, self).__init__()
        self.name = "torch_sgns"
        self._corpus = corpus
        self._embedding_dim = embedding_dim
        self._vocab_size = corpus.vocab_size
        self._save_path = os.path.join("./data/", "{}_weights.pth".format(self.name))

        self.embeddings_center = nn.Embedding(corpus.vocab_size, embedding_dim)
        self.embeddings_outside = nn.Embedding(corpus.vocab_size, embedding_dim)

    def forward(self, center_ids, outside_ids, neg_ids):
        """
        forward computation graph for the model

        :param center_ids: center word indices in the batch
        :type center_ids: List -> len(center_ids) = batch size
        :param outside_ids: outside (context) word indices in the batch
        :type outside_ids: List -> len(center_ids) = batch size
        :param neg_ids: negative words sampled for training
        :type neg_ids: List[List]
        :return: loss for the current batch
        :rtype: scalar
        """
        # get batch size
        bs = len(center_ids)
        # let b -> batch size, d-> embedding_dim
        # get center word vector (b, d)
        v_c = self.embeddings_center(center_ids)
        # get outside word vector (b, d)
        u_o = self.embeddings_outside(outside_ids)
        # take dot product
        pos_dot = torch.sum(torch.mul(v_c, u_o), dim=1)  # (b, )
        pos_out = torch.sum(F.logsigmoid(pos_dot))  # scalar

        # get negative words embeddings (b, ng, d)
        # where ng -> num negative words per example
        V_k = self.embeddings_outside(neg_ids)
        # batch matrix multiplication
        # (b, ng, d) @ (b, d, 1) - > (b, ng, 1)
        neg_dot = torch.sum(torch.bmm(V_k, v_c.unsqueeze(2)).squeeze(-1), dim=1)  # (b)
        neg_out = torch.sum(F.logsigmoid(-neg_dot))  # scalar
        loss = -(pos_out + neg_out) / bs  # similar to NCE loss
        return loss

    def build_embeddings(self):
        """
        executes training of the model and builds word vector representations
        """
        self._batch_generator = BatchGenerator(self._corpus).generate_batch(
            neg_sampling=True
        )

        loss_history = self.train()
        plot_loss(loss_history, self.name)

    def train(self, alpha=0.025, max_iter=3000):
        """
        training of sgns model

        :param alpha: initial learning rate, defaults to 0.01
        :type alpha: float, optional
        :param max_iter: maximun number of iterations, defaults to 5000
        :type max_iter: int, optional
        :return: loss history
        :rtype: List
        """
        start_time = time()
        print_interval = 10
        save_interval = 500
        anneal_interval = 1000

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        loss_history = []

        # training loop
        for i in range(1, max_iter):
            # zero the gradients for new batch
            self.zero_grad()

            # get training data
            center_ids, outside_ids, negative_ids = next(self._batch_generator)
            center_ids = torch.tensor(center_ids)
            outside_ids = torch.tensor(outside_ids)
            negative_ids = torch.tensor(negative_ids)

            # compute loss
            loss = self(center_ids, outside_ids, negative_ids)
            # compute gradients
            loss.backward()
            # update parameters
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

            self.optimizer.step()
            # track loss
            loss_history.append(loss.item())

            if (i % print_interval) == 0:
                ts, ls = (time() - start_time) / 60.0, loss.item()
                logger.info(
                    "Iter: {} | Loss: {:.2f} | Time: {:.1f} mins".format(i, ls, ts)
                )
                plot_loss(loss_history, self.name)

            if (i % anneal_interval) == 0:
                alpha /= 2.0
                logger.info("Changing learing rate to {}".format(alpha))
                for g in self.optimizer.param_groups:
                    g["lr"] = alpha

            if (i % save_interval) == 0:
                self.save_model()

        return loss_history

    @property
    def word_embeddings(self):
        """
        produce final word embeddings as average of center and outside vectors
        embeddings are L2 normalized

        :return: word embeddings
        :rtype: np.ndarray
        """
        with torch.no_grad():
            mat_center = self.embeddings_center.weight
            mat_outside = self.embeddings_outside.weight
            word_vecs = (mat_center + mat_outside) * 0.5
            word_vecs = F.normalize(word_vecs, p=2, dim=1)
        return word_vecs.numpy()

    def get_embeddings(self, words):
        """
        get embeddings for query words

        :param words: query words
        :type words: List
        :return: word embeddings
        :rtype: np.ndarray
        """
        assert isinstance(words, list), "parameter words must be a list"
        emb = self.word_embeddings
        indices = [self._corpus.word_to_idx.get(word, "UNK") for word in words]
        these_vecs = emb[indices]
        assert these_vecs.shape == (len(words), self._embedding_dim)
        return these_vecs

    def get_neighbors(self, query, top_k=5):
        """
        get top_k most similar words for the query

        :param query: query word
        :type query: str
        :param top_k: size of the neighbor, defaults to 5
        :type top_k: int, optional
        :return: neighboring words
        :rtype: List
        """
        assert query in self._corpus.vocab, "out of vocabulary word used as query"
        emb = self.word_embeddings
        query_idx = self._corpus.word_to_idx[query]
        query_vec = emb[query_idx].reshape(-1, 1)  # (d,1)
        assert query_vec.shape == (self._embedding_dim, 1)

        scores = np.dot(emb, query_vec).squeeze()
        ranks = np.argsort(scores)[::-1]
        neighbors = [self._corpus.idx_to_word[idx] for idx in ranks if idx != query_idx]
        return neighbors[:top_k]

    def save_model(self):
        """
        save the trained model
        """
        torch.save(self.state_dict(), self._save_path)

    def load_model(self):
        """
        load the trained model
        """
        if not Path(self._save_path).is_file():
            raise FileNotFoundError
        self.load_state_dict(torch.load(self._save_path))


if __name__ == "__main__":
    df = pd.read_pickle("./data/processed_dataset.pkl")
    corpus = ArticleCorpus.from_dataframe(df, "abstract")

    # word2vec = Word2VecBase(loss_type="nce")
    # word2vec.create_embeddings(corpus)
    # print(
    # np.dot(
    # word2vec.get_embeddings(["micromorphic"])[0],
    # word2vec.get_embeddings(["continuum"])[0],
    # )
    # )

    model = Word2VecTorch(corpus)
    model.build_embeddings()
    print(model.get_neighbors("micromorphic"))

    print(model)
