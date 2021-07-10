from abc import ABCMeta, abstractmethod


class IWordEmbedding(metaclass=ABCMeta):
    """
    An interface to create concrete word embedding classes
    """

    @abstractmethod
    def build_embeddings(self, corpus):
        """
        abstract method enforcing training of word embedding functionality

        :param corpus: word corpus object represing training dataset
        :type corpus: ArticleCorpus
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, words):
        """
        abstract method enforcing word embedding functionality

        :param words: a list of words for which embeddings are requested
        :type word: List[str]
        """
        raise NotImplementedError
