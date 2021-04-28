import datetime
import os

import numpy as np
import pandas as pd
from loguru import logger


class TripDataCleansing:
    """Load, transform, and Dump trip data"""

    def __init__(self, raw_local, target_local):
        self.raw_local = raw_local
        self.target_local = target_local

        self.__prepare_config()

    def __prepare_config(self):
        """calculate other useful configs"""

        try:
            self.raw_data_files = os.listdir(self.raw_local)
            # get the csv data files
            self.raw_data_files = [
                os.path.join(self.raw_local, i)
                for i in self.raw_data_files
                if i.endswith(".csv")
            ]
        except Exception as ee:
            raise Exception(f"Could not listdir for {self.raw_local}\n{ee}")

    def _load_all_trip_data(self):
        """Load all trip data from config data path"""

        # Load all csv files into dataframes
        df_array = [pd.read_csv(csv_file) for csv_file in self.raw_data_files]
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
        self.trip_data = df_all

    def _datetime_transformations(self):
        """Standardize datetime formats"""

        # extract date from datetime strings
        # they have different formats for dates so it is easier to
        # use pandas
        self.trip_data["date"] = self.trip_data.start_time.apply(
            lambda x: x.split(" ")[0] if x else None
        )
        self.trip_data["date"] = pd.to_datetime(self.trip_data.date)

        # extract hour of the day
        # there exists different time formats
        self.trip_data["hour"] = self.trip_data.start_time.apply(
            lambda x: int(float(x.split(" ")[-1].split(":")[0]))
        )

        # get weekday
        self.trip_data["weekday"] = self.trip_data.date.apply(lambda x: x.weekday())

        # get month
        self.trip_data["month"] = self.trip_data.date.apply(lambda x: x.month)

    def _duration_normalization(self):
        """Duration was recorded as seconds before 2017-04-01.

        Here we will normalized durations to minutes
        """

        df_all_before_2017q1 = self.trip_data.loc[
            self.trip_data.date < pd.to_datetime(datetime.date(2017, 4, 1))
        ]
        df_all_after_2017q1 = self.trip_data.loc[
            self.trip_data.date >= pd.to_datetime(datetime.date(2017, 4, 1))
        ]

        df_all_before_2017q1["duration"] = df_all_before_2017q1.duration / 60

        self.trip_data = pd.concat([df_all_before_2017q1, df_all_after_2017q1])

    def _fill_station_id(self):
        """start_station_id has null values

        fillna with 0 for the station id
        """
        self.trip_data["start_station_id"].fillna(0, inplace=True)

    def _backfill_bike_types(self):
        """Bike types did not exist until q3 of 2018
        because they only had standard before this.
        """

        self.trip_data["bike_type"] = self.trip_data.bike_type.fillna("standard")

    def _normalize_coordinates(self):
        """Bike coordinates have diverging types: str or float, normalizing to float"""

        def convert_to_float(data):
            try:
                return float(data)
            except Exception:
                logger.debug(f"Can not convert {data}")
                return np.nan

        self.trip_data["start_lat"] = self.trip_data.start_lat.apply(convert_to_float)
        self.trip_data["start_lon"] = self.trip_data.start_lon.apply(convert_to_float)
        self.trip_data["end_lat"] = self.trip_data.end_lat.apply(convert_to_float)
        self.trip_data["end_lon"] = self.trip_data.end_lon.apply(convert_to_float)

    def _normalized_bike_id(self):
        """
        _normalized_bike_id bike_id can be str or int or float

        not all ids can be converted to int so we will use str
        """

        self.trip_data["bike_id"] = self.trip_data.bike_id.apply(str)

    def _save_all_trip_data(self):
        """Dump all trip data to the destination define in config"""

        logger.debug(self.trip_data.sample(10))

        try:
            if self.target_local.endswith(".parquet"):
                self.trip_data.to_parquet(self.target_local, index=False)
            elif self.target_local.endswith(".csv"):
                self.trip_data.to_csv(self.target_local, index=False)
            else:
                raise ValueError(
                    f"Specified target_local is not valid (should be .csv or .parquet):{self.target_local}"
                )
        # TODO: should be more specific about the exceptions
        except Exception as ee:
            raise Exception(f"Could not save data to {self.target_local}")

    def _load_station_info(self):
        """TODO: write the function to load station info"""

        return

    def _load_station_status(self):
        """TODO: write the function to load station status"""

        return

    def pipeline(self):
        """Connect the pipes of data operations"""

        # load all csv files to one dataframe
        self._load_all_trip_data()

        # Transformations
        self._datetime_transformations()
        self._duration_normalization()
        self._backfill_bike_types()
        self._fill_station_id()
        self._normalize_coordinates()
        self._normalized_bike_id()

        # dave data
        self._save_all_trip_data()

        return {"data_file": self.target_local}


if __name__ == "__main__":
    cleaner_config = {}
    cleaner = TripDataCleansing(cleaner_config)
    cleaner.pipeline()

    print("END OF GAME")
