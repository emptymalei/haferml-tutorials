import datetime
import os

import numpy as np
import pandas as pd
from loguru import logger
from haferml.preprocess.pipeline import BasePreProcessor, attributes


def get_all_raw_files(raw_local):
    """Get the list of data files
    """
    raw_data_files = os.listdir(raw_local)
    raw_data_files = [
        os.path.join(raw_local, i)
        for i in raw_data_files
        if i.endswith(".csv")
    ]

    return raw_data_files


def load_data(raw_data_files):
    """Load all the raw datasets
    """
    # Load all csv files into dataframes
    df_array = [pd.read_csv(csv_file) for csv_file in raw_data_files]
    # Rename *_station to *_station_id
    # they have two different types of column names for stations
    df_array = [
        df_temp.rename(
            columns={
                "end_station": "end_station_id",
                "start_station": "start_station_id",
            }
        )
        for df_temp in df_array
    ]

    # combine all dataframe
    # beware of missing data: bike_type
    # bike_type was added in 2018 q3
    df_all = pd.concat(df_array)

    # link pointer to class attributes
    return df_all


class TripDataCleansing(BasePreProcessor):
    """Load, transform, and Dump trip data"""

    def __init__(self, config, **params):
        super(TripDataCleansing, self).__init__(
            config=config,
            **params
        )

    @attributes(order=1)
    def _datetime_transformations(self, dataframe):
        """Standardize datetime formats"""

        # extract date from datetime strings
        # they have different formats for dates so it is easier to
        # use pandas
        self.dataframe = dataframe
        self.dataframe["date"] = self.dataframe.start_time.apply(
            lambda x: x.split(" ")[0] if x else None
        )
        self.dataframe["date"] = pd.to_datetime(self.dataframe.date)

        # extract hour of the day
        # there exists different time formats
        self.dataframe["hour"] = self.dataframe.start_time.apply(
            lambda x: int(float(x.split(" ")[-1].split(":")[0]))
        )

        # get weekday
        self.dataframe["weekday"] = self.dataframe.date.apply(lambda x: x.weekday())

        # get month
        self.dataframe["month"] = self.dataframe.date.apply(lambda x: x.month)

    @attributes(order=2)
    def _duration_normalization(self, dataframe):
        """Duration was recorded as seconds before 2017-04-01.

        Here we will normalized durations to minutes
        """

        df_all_before_2017q1 = self.dataframe.loc[
            self.dataframe.date < pd.to_datetime(datetime.date(2017, 4, 1))
        ]
        df_all_after_2017q1 = self.dataframe.loc[
            self.dataframe.date >= pd.to_datetime(datetime.date(2017, 4, 1))
        ]

        df_all_before_2017q1["duration"] = df_all_before_2017q1.duration / 60

        self.dataframe = pd.concat([df_all_before_2017q1, df_all_after_2017q1])

    @attributes(order=3)
    def _backfill_bike_types(self, dataframe):
        """Bike types did not exist until q3 of 2018
        because they only had standard before this.
        """

        self.dataframe["bike_type"] = self.dataframe.bike_type.fillna("standard")

    @attributes(order=4)
    def _fill_station_id(self, dataframe):
        """start_station_id has null values

        fillna with 0 for the station id
        """
        self.dataframe["start_station_id"].fillna(0, inplace=True)

    @attributes(order=5)
    def _normalize_coordinates(self, dataframe):
        """Bike coordinates have diverging types: str or float, normalizing to float"""

        def convert_to_float(data):
            try:
                return float(data)
            except Exception:
                logger.debug(f"Can not convert {data}")
                return np.nan

        self.dataframe["start_lat"] = self.dataframe.start_lat.apply(convert_to_float)
        self.dataframe["start_lon"] = self.dataframe.start_lon.apply(convert_to_float)
        self.dataframe["end_lat"] = self.dataframe.end_lat.apply(convert_to_float)
        self.dataframe["end_lon"] = self.dataframe.end_lon.apply(convert_to_float)

    @attributes(order=6)
    def _normalized_bike_id(self, dataframe):
        """
        _normalized_bike_id bike_id can be str or int or float

        not all ids can be converted to int so we will use str
        """

        self.dataframe["bike_id"] = self.dataframe.bike_id.apply(str)

    @attributes(order=7)
    def _save_all_trip_data(self, dataframe):
        """Dump all trip data to the destination define in config"""

        logger.debug(self.dataframe.sample(10))
        self.target_local = self.params["target_local"]

        try:
            if self.target_local.endswith(".parquet"):
                self.dataframe.to_parquet(self.target_local, index=False)
            elif self.target_local.endswith(".csv"):
                self.dataframe.to_csv(self.target_local, index=False)
            else:
                raise ValueError(
                    f"Specified target_local is not valid (should be .csv or .parquet):{self.target_local}"
                )
        # TODO: should be more specific about the exceptions
        except Exception as ee:
            raise Exception(f"Could not save data to {self.target_local}")


if __name__ == "__main__":
    cleaner_config = {}
    cleaner = TripDataCleansing(cleaner_config)
    cleaner.pipeline()

    print("END OF GAME")
