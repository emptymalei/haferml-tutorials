import os

import click
import pandas as pd
import simplejson as json
from dotenv import load_dotenv
from haferml.blend.config import Config
from haferml.sync.local import prepare_folders
from loguru import logger

from utils.transformer import TripDataCleansing
from utils.transformer import get_all_raw_files
from utils.transformer import load_data

load_dotenv()


@click.command()
@click.option(
    "-c",
    "--config",
    type=str,
    default=os.getenv("CONFIG_FILE"),
    help="Path to config file",
)
def transform(config):

    base_folder = os.getenv("BASE_FOLDER")

    _CONFIG = Config(config, base_folder=base_folder)

    etl_trip_data_config = _CONFIG[["etl", "raw", "trip_data"]]
    transformed_trip_data_config = _CONFIG[["etl", "transformed", "trip_data"]]

    # create folders
    prepare_folders(etl_trip_data_config["local"], base_folder)
    prepare_folders(transformed_trip_data_config["local"], base_folder)

    # load raw data
    raw_data_files = get_all_raw_files(etl_trip_data_config["local_absolute"])
    dataset = load_data(raw_data_files)

    # data cleansing
    logger.info("Cleaning up data")
    cleaner = TripDataCleansing(
        config=_CONFIG,
        target_local=transformed_trip_data_config["name_absolute"],
    )
    cleaner.run(dataset)
    logger.info("Saved clean data: {}".format(cleaner.target_local))


if __name__ == "__main__":
    transform()
    print("END OF GAME")
