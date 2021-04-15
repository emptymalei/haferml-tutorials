import os

import click
import simplejson as json
from dotenv import load_dotenv
from loguru import logger

from utils.fetch import DataDownloader, DataLinkHTMLExtractor
from utils.fetch import get_page_html as _get_page_html

load_dotenv()


def load_config(config_path):
    """
    load_config loads the config files of the project

    :param config_path: path to the config file
    :type config_path: str, optional
    """

    if not os.path.exists(config_path):
        raise Exception(
            f"config file path {config_path} does not exist! Beware of the relative path."
        )

    logger.debug(f"Loading config from {config_path}")
    with open(config_path, "r") as fp:
        config = json.load(fp)

    if not config:
        logger.warning(f"The config is empty: {config}")

    return config


def get_config(configs, path):
    """
    Get value of the configs under specified path

    :param dict configs: input dictionary
    :param list path: path to the value to be obtained

    >>> get_config({'etl':{'raw':{'local':'data/raw', 'remote': 's3://haferml-tutorials/rideindego/marshall/data/raw'}}},['etl','raw'])
    {'local':'data/raw', 'remote': 's3://haferml-tutorials/rideindego/marshall/data/raw'}
    """

    # Construct the path
    if not isinstance(path, (list, tuple)):
        logger.warning(f"path is not list nor tuple, converting to list: {path}")
        path = [path]

    # Find the values
    res = configs.copy()
    for p in path:
        res = res[p]

    return res


def construct_paths(config, base_folder=os.getenv("BASE_FOLDER")):
    """
    construct_paths reconstructs the path based on base folder
    """

    if not config.get("local"):
        raise Exception(f"{config} does not contain local key ")

    config_recon = {}
    config_local = config["local"]
    config_name = config.get("name")
    config_local = os.path.join(
        base_folder,
        config_local
    )
    config_recon["local"] = config_local
    if config_name:
        config_local_full = os.path.join(
            config_local,
            config_name
        )
        config_recon["file_path"] = config_local_full

    return {
        **config,
        **config_recon
    }



@click.command()
@click.option(
    "-c", "--config", type=click.Path(exists=True),
    default=os.getenv("CONFIG_FILE"),
    help="Path to config file"
)
def extract(config):

    base_folder = os.getenv("BASE_FOLDER")

    _CONFIG = load_config(config)

    etl_trip_data_config = get_config(_CONFIG, ["etl", "raw", "trip_data"])
    etl_trip_data_config = construct_paths(etl_trip_data_config, base_folder)
    logger.info(f"Using config: {etl_trip_data_config}")
    # create folders
    if not os.path.exists(etl_trip_data_config["local"]):
        os.makedirs(etl_trip_data_config["local"])

    # Download Raw Data
    source_link = etl_trip_data_config["source"]
    logger.info(f"Will download from {source_link}")
    page = _get_page_html(source_link).get("data", {})
    page_extractor = DataLinkHTMLExtractor(page)
    links = page_extractor.get_data_links()
    logger.info(f"Extracted links from {source_link}: {links}")

    # Download data
    dld = DataDownloader(
        links, data_type="zip",
        folder=etl_trip_data_config["local"]
    )
    dld.run()


if __name__ == "__main__":
    extract()
    print("END OF GAME")
