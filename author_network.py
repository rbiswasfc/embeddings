import logging
import itertools
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter
from utils import preprocess_fn, setup_logger, try_except
from networkx.algorithms.community.modularity_max import greedy_modularity_communities


logger = logging.getLogger(__name__)
logger = setup_logger(logger)


class AuthorNetwork(object):
    """
    A class to represent network of authors
    """

    def __init__(self):
        """
        initialize and build the network
        """
        self.network = None
        self.communities = None
        self.authors = 0

        self.read_dataset()
        self.get_author_clusters()
        self.get_authors()
        self.get_author_article_counts()
        self.build_network()

    def read_dataset(self):
        """
        load dataset to be used in the network
        """
        self.data = pd.read_pickle("./data/processed_dataset.pkl")

    def get_author_clusters(self):
        """
        get author clusters. The list of authors of an article constitutes a cluster. 
        """
        author_clusters = []
        for _, row in self.data.iterrows():
            these_authors = row.authors
            author_clusters.append(these_authors)
            citations = row.cited_by
            if len(citations) > 0:
                for _, v in citations.items():
                    cited_authors = [preprocess_fn(auth) for auth in v["authors"]]
                    author_clusters.append(cited_authors)
        self.author_clusters = author_clusters

    def get_author_article_counts(self):
        """
        create a dict which contains number of articles published by each author in the dataset
        """
        flattened_authors = list(itertools.chain(*self.author_clusters))
        author_counts = Counter(flattened_authors)  # .most_common()
        self.author_to_num_articles = author_counts

    def get_authors(self):
        """
        get the list of authors in the dataset
        """
        flattened_authors = list(itertools.chain(*self.author_clusters))
        self.authors = list(set(flattened_authors))

    def build_network(self):
        """
        build the author network using networkx
        """
        G = nx.Graph()
        G.add_nodes_from(self.authors)

        # add edges
        list_edges = []
        for this_cluster in self.author_clusters:
            if len(this_cluster) < 10:
                combs = list(itertools.combinations(this_cluster, 2))
                for this_comb in combs:
                    list_edges.append((this_comb[0], this_comb[1]))
        G.add_edges_from(list_edges)
        self.network = G
        self.num_nodes, self.num_edges = G.number_of_nodes(), G.number_of_edges()
        logger.info(
            "Author Network is built with {} nodes and {} edges".format(
                self.num_nodes, self.num_edges
            )
        )

    def find_neighbor(self, author, order=1):
        """
        find neighbor of a author. If order is 1, this would be all co-authors of the author.
        If order=2, the neighbor will include all co-authors of co-authors of the author.

        :param author: author under consideration
        :type author: str
        :param order: number of hops to consider, defaults to 1
        :type order: int, optional
        :return: a list of neighboring author names
        :rtype: List[str]
        """
        if author not in self.authors:
            logger.info("Error: Author not found!")
            return None
        group = set([author])
        for proximity in range(order):
            group = set((nbr for member in group for nbr in self.network[member]))
        return list(group)

    @try_except(logger)
    def find_shortest_paths(self, source, target):
        """
        all all possible paths from source to target author with minimum number of hops.

        :param source: source author
        :type source: str
        :param target: target author
        :type target: str
        :return: a list containing linking authors from source to target
        :rtype: List[str]
        """
        paths = nx.all_shortest_paths(self.network, source=source, target=target)
        return list(paths)

    def plot_subnetwork(self, subset_nodes):
        """
        visualize a network with a subset of authors

        :param subset_nodes: a list of authors under consideration
        :type subset_nodes: List[str]
        """
        SG = self.network.subgraph(subset_nodes)
        pos = nx.circular_layout(SG)
        nx.draw(
            SG,
            with_labels=True,
            node_color="skyblue",
            pos=pos,
            node_size=500,
            font_size=18,
        )
        plt.savefig("./data/sub_graph.png", dpi=200)
        plt.clf()

    def get_network_stats(self):
        """
        get stats of the author network
        """
        # density
        self.density = round(nx.density(self.network), 5)
        # degree
        degree = self.network.degree()
        degree_list = []
        for (n, d) in degree:
            degree_list.append(d)
        self.ave_degree = sum(degree_list) / len(degree_list)

        # clustering coefficient
        local_coeffs = nx.algorithms.cluster.clustering(self.network)
        self.ave_clustering_coeff = round(
            sum(local_coeffs.values()) / len(local_coeffs), 5
        )

        # output
        logger.info("Author Network Density = {}".format(self.density))
        logger.info("Author Network Average Degree = {}".format(self.ave_degree))
        logger.info(
            "Author Network Average Clustering Coefficient = {}".format(
                self.ave_clustering_coeff
            )
        )

    def create_communities(self):
        """
        create communities within the network. All members of a given community are closely related.
        """
        author_com = greedy_modularity_communities(self.network)
        self.communities = list(author_com)
        self.num_communities = len(self.communities)
        logger.info("{} communities created.".format(self.num_communities))

    def get_community_members(self, community_idx):
        """
        Find members of a given community

        :param community_idx: community index
        :type community_idx: int
        :return: members of the community
        :rtype: List[str]
        """
        if not self.communities:
            self.create_communities()
        if community_idx >= self.num_communities:
            logger.error("Community index is out of bound...")
            return None
        members = list(self.communities[community_idx])
        return members

    def get_community_topics(self, community_idx):
        """
        get community research interests

        :param community_idx: community index
        :type community_idx: int
        :return: a list of key words of research topics attributed to the community
        :rtype: List[str]
        """
        members = self.get_community_members(community_idx)
        flags = (
            self.data["authors"]
            .apply(lambda x: True if set(x).intersection(set(members)) else False)
            .values
        )
        tmp_df = self.data[flags].copy()
        kws = Counter(list(itertools.chain(*tmp_df.author_kws.values))).most_common(10)
        topics = []
        for word, _ in kws:
            topics.append(word)
        return topics

    def __repr__(self):
        return "A Network of {} Authors".format(len(self.authors))


if __name__ == "__main__":
    an = AuthorNetwork()
    print(an.find_neighbor("biswas r"))
    paths = an.find_shortest_paths("biswas r", "leclerc j")
    for path in paths:
        print(path)

    an.get_network_stats()
    members = an.get_community_members(community_idx=35)
    print(members)
    an.plot_subnetwork(members)

    # topics = an.get_community_topics(community_idx=25)
    # print(topics)

    topics = an.get_community_topics(community_idx=35)
    print(topics)
