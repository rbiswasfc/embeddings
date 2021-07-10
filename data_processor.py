import glob
import logging
import pandas as pd
from ast import literal_eval
from utils import setup_logger, try_except, preprocess_fn, get_dataframe_memory_usage


logger = logging.getLogger(__name__)
logger = setup_logger(logger)


@try_except(logger)
def convert_value(x):
    """
    helper function to infer appropriate type of data from external sources

    :param x: input data
    :type x: str
    :return: converted data
    :rtype: dynamically set by literal eval
    """
    return literal_eval(x)


def process_dataframe_from_csv(df, cols):
    """
    Process a dataframe loaded from csv file. It will convert columns to proper data type

    :param df: input dataframe
    :type df: pd.DataFrame
    :param cols: columns to be processed
    :type cols: list
    """

    for col in cols:
        df[col] = df[col].apply(convert_value)
    return df


def prepare_dataset():
    """
    prepare dataset for the current project

    :return: dataframe containing relevant info
    :rtype: pd.DataFrame
    """
    # load article data
    df_list = []
    for file_path in glob.glob("./data/scrapper_*.csv"):
        df = pd.read_csv(file_path)
        df_list.append(df)
    df_data = (
        pd.concat(df_list).drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    )

    # load metadata obtained from scopus
    df_list = []
    for file_path in glob.glob("./data/scopus_*.csv"):
        df = pd.read_csv(file_path)
        df_list.append(df)
    df_meta = (
        pd.concat(df_list).drop_duplicates(subset=["scopus_id"]).reset_index(drop=True)
    )

    # keep only relevant columns
    df_meta = df_meta[
        [
            "scopus_id",
            "first_author",
            "affiliation",
            "title",
            "publication_name",
            "publication_date",
            "citation_count",
        ]
    ].copy()

    df_data = df_data[
        ["article_id", "authors", "abstract", "author_kws", "index_kws", "cited_by"]
    ].copy()

    df_data = df_data.rename(columns={"article_id": "scopus_id"})

    # combine datasources
    df_focus = pd.merge(df_meta, df_data, how="inner", on="scopus_id")

    # handling of missing values
    df_focus = df_focus[~df_focus["authors"].isna()].copy()

    # cast columns with proper data types
    df_focus = process_dataframe_from_csv(
        df_focus, ["affiliation", "authors", "author_kws", "index_kws", "cited_by"]
    )

    # casting missing values
    df_focus["author_kws"] = df_focus["author_kws"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df_focus["index_kws"] = df_focus["index_kws"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df_focus["cited_by"] = df_focus["cited_by"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    # pre-processing
    df_focus["first_author"] = df_focus["first_author"].apply(preprocess_fn)
    df_focus["title"] = df_focus["title"].apply(preprocess_fn)
    df_focus["abstract"] = df_focus["abstract"].apply(preprocess_fn)
    df_focus["affiliation"] = df_focus["affiliation"].apply(
        lambda x: [preprocess_fn(elem) for elem in x]
    )
    df_focus["authors"] = df_focus["authors"].apply(
        lambda x: [preprocess_fn(elem) for elem in x]
    )
    df_focus["author_kws"] = df_focus["author_kws"].apply(
        lambda x: [preprocess_fn(elem) for elem in x]
    )
    df_focus["index_kws"] = df_focus["index_kws"].apply(
        lambda x: [preprocess_fn(elem) for elem in x]
    )

    df_focus["publication_date"] = pd.to_datetime(
        df_focus["publication_date"], errors="coerce"
    ).fillna(pd.Timestamp.max)
    logger.info(get_dataframe_memory_usage(df_focus))
    df_focus.to_pickle("./data/processed_dataset.pkl")

    return df_focus


if __name__ == "__main__":
    df = prepare_dataset()
