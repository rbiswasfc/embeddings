from word2vec_embeddings import Word2Vec


class EmbeddingFactory:
    """
    factory class for word embeddings
    """

    @staticmethod
    def get_embedding_producer(method):
        """
        a static method to get word embedding producer object 
        each producer object utilizes a specific word embedding algorithm
        e.g. word2vec, glove, bert, co-occurrance 
        """
        supported_algos = ["bert", "word2vec", "glove", "co_occurrance"]

        if method.lower() == "word2vec":
            return Word2Vec()


if __name__ == "__main__":
    pass
