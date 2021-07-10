import json
import uuid
import logging
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from scraper_api import ScraperAPIClient
from utils import try_except, setup_logger

logger = logging.getLogger(__name__)
logger = setup_logger(logger)


def get_article_metadata(client, query, start_year=2015, filepath="data/temp.csv"):
    """ 
    Fetch article metadata matching a search query from scopus
    :param client: represents a client interface to api.elsevier.com
    :type client: ElsClient
    :param query: search query e.g. "TITLE-ABS-KEY ( micromorphic AND homogenization )"
    :type query: str
    :param start_year: start year to search articles, defaults to 2015
    :type start_year: int, optional
    :param filepath: path to the file where results will be stored, defaults to 'data/temp.json'
    :type filepath: str, optional

    :return dataframe with retrieved article details
    """
    query += "AND PUBYEAR > {}".format(str(start_year))
    logger.info("Fetching metadata with query {}".format(query))
    doc_search = ElsSearch(query, "scopus")
    doc_search.execute(client, get_all=False)
    if len(doc_search.results) <= 1:
        logger.warning("No result found")
        return pd.DataFrame()
    logger.info("# of article retrieved: {}".format(len(doc_search.results)))

    article_dict = dict()
    for idx, article in enumerate(doc_search.results):
        try:
            tmp_dict = dict()
            tmp_dict["scopus_id"] = int(article["prism:url"].split("/")[-1])
            tmp_dict["title"] = article["dc:title"]
            tmp_dict["first_author"] = article["dc:creator"]
            tmp_dict["affiliation"] = [
                item["affilname"] for item in article["affiliation"]
            ]
            tmp_dict["publication_name"] = article["prism:publicationName"]
            tmp_dict["publication_date"] = article["prism:coverDate"]
            tmp_dict["citation_count"] = article["citedby-count"]
            for link in article["link"]:
                if link["@ref"] == "scopus":
                    tmp_dict["abstract_link"] = link["@href"]
                elif link["@ref"] == "scopus-citedby":
                    tmp_dict["cited_by_link"] = link["@href"]
            article_dict[idx] = tmp_dict
        except Exception:
            logger.exception("Exception article parsing...")
            logger.debug(article)

    df = pd.DataFrame(article_dict).T
    df.to_csv(filepath, index=False)
    return df


class ArticleDataExtractor:
    """
    A class to facilitate extracion of scientific publication data from scopus
    """

    def __init__(self, client):
        """
        initialization for the class

        :param client: a ScraperAPIClient
        :type client: ScraperAPIClient
        """
        self.client = client
        self.soup = BeautifulSoup()

    @try_except(logger)
    def fetch_data(self, url):
        """
        fetch data from the web using scraper api

        :param url: url from which data will be fetched
        :type url: str
        :return: response code
        :rtype: int
        """
        result = self.client.get(url=url, render=False, autoparse=True)
        self.soup = BeautifulSoup(result.text)
        return result.status_code

    @try_except(logger)
    def _get_title(self):
        """
        Get article title

        :return: article title
        :rtype: str
        """
        return self.soup.title.text.split("Document details -")[-1].strip()

    @try_except(logger)
    def _get_abstract(self):
        """
        Get article abstract

        :return: abstract
        :rtype: str
        """
        tmp = self.soup.find("section", id="abstractSection")
        abstract = "".join([para.text.strip() for para in tmp.findAll("p")])
        return abstract

    @try_except(logger)
    def _get_authors(self):
        """
        Get article authors

        :return: a list of authors
        :rtype: List
        """
        tmp = self.soup.find("section", id="authorlist")
        authors = [
            auth.text.strip() for auth in tmp.findAll("span", class_="previewTxt")
        ]
        return authors

    @try_except(logger)
    def _get_author_keywords(self):
        """
        Get article keywords as specified by authors

        :return: keyword list
        :rtype: List
        """
        tmp = self.soup.find("section", id="authorKeywords")
        author_kws = [kw.text.strip() for kw in tmp.findAll("span")]
        return author_kws

    @try_except(logger)
    def _get_index_keywords(self):
        """
        Get article keywords as specified by scopus

        :return: keyword list
        :rtype: List
        """
        tmp = self.soup.find("section", id="indexedKeywords")
        index_kws = [kw.text.strip() for kw in tmp.findAll("span", class_="badges")]
        return index_kws

    @try_except(logger)
    def _get_citedby_articles(self):
        """
        Get a sample of recent articles which cites the current article

        :return: info on citedby articles
        :rtype: dict
        """
        tmp = self.soup.find("div", id="recordPageBoxes")
        citedby_dict = dict()
        for idx, kw in enumerate(tmp.findAll("div", class_="recordPageBoxItem")):
            authors = []
            for auth in kw.findAll(
                "span",
                title="Search for all documents by this author: Subscription required",
            ):
                authors.append(auth.text.strip())
            title = kw.find("span", class_="authorLinks").text.strip()
            journal = kw.find("span", class_="italicText").text.strip()
            citedby_dict[idx] = {"authors": authors, "title": title, "journal": journal}
        return citedby_dict

    def execute(self, url):
        """
        orchestrator method for extracting the article information 

        :param url: url from which data will be extracted
        :type url: str
        :return: a dict with article information
        :rtype: dict
        """
        status_code = self.fetch_data(url)
        if status_code != 200:
            print("Error in fetching data from Scopus! Exiting")
            return dict()

        article = dict()
        article["title"] = self._get_title()
        article["authors"] = self._get_authors()
        article["abstract"] = self._get_abstract()
        article["author_kws"] = self._get_author_keywords()
        article["index_kws"] = self._get_index_keywords()
        article["cited_by"] = self._get_citedby_articles()

        return article


def fetch_dataset():
    """
    prepare dataset of academic articles

    :return: dataset
    :rtype: pd.DataFrame
    """
    with open("config.json") as f:
        config = json.load(f)

    ## Initialize clients
    elsevier_client = ElsClient(config["elsevier_apikey"])
    scraper_client = ScraperAPIClient(config["scraper_apikey"])
    file_uuid = uuid.uuid4().hex

    # search_queries = [
    #     "TITLE-ABS-KEY ( localizing  AND gradient  AND damage  AND model )",  # 21 results
    #     "TITLE-ABS-KEY ( micromorphic  AND computational ) ",  # 58 results
    #     "TITLE-ABS-KEY ( ultra  AND high  AND performance  AND concrete  AND projectile  AND impact )",  # 76 articles
    # ]
    search_queries = config["queries"]

    # get metadata
    logger.info("=" * 30)
    logger.info("Stage 1 >>>>>")
    logger.info("Extacting metadata....")
    meta_list = []
    for query in search_queries:
        this_df = get_article_metadata(elsevier_client, query)
        # logger.info(this_df.head())
        meta_list.append(this_df)
    df_metadata = pd.concat(meta_list).reset_index()
    df_metadata = df_metadata.drop_duplicates(subset=["scopus_id"])
    logger.info("# of articles found = {}".format(len(df_metadata)))
    logger.info("Metadata extraction completed.")
    logger.info("=" * 30)

    # get additional data
    max_calls = 3000
    num_sample = min(max_calls, len(df_metadata))
    df_metadata = df_metadata.sample(num_sample)
    df_metadata.to_csv("data/scopus_metadata_{}.csv".format(file_uuid), index=False)
    logger.info("Stage 2 >>>>>>")
    logger.info("Extracting additional data...")
    all_articles = dict()
    data_extractor = ArticleDataExtractor(scraper_client)

    batch_size = 10
    n_batch = int(len(df_metadata) / batch_size) + 1
    for i in tqdm(range(n_batch)):
        start, end = i * batch_size, (i + 1) * batch_size
        tmp_df = df_metadata.iloc[start:end].copy()

        for _, row in tmp_df.iterrows():
            article_id, article_url = row.scopus_id, row.abstract_link
            logger.debug("Extracting info from {}".format(article_url))
            this_article = data_extractor.execute(article_url)
            this_article["article_id"] = article_id
            all_articles[uuid.uuid4().hex] = this_article
        df_data = pd.DataFrame(all_articles).T.reset_index(drop=True)
        # save intermediate result
        file_path = "data/scrapper_data_{}.csv".format(file_uuid)
        df_data.to_csv(file_path, index=False)

    logger.info("Stage 2 complete")
    logger.info("=" * 30)

    return df_metadata, df_data


if __name__ == "__main__":
    df_1, df_2 = fetch_dataset()

