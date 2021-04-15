import datetime
from logging import log
import os

import click
import joblib
import pandas as pd
import simplejson as json
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import category_encoders as ce

logger.info(f"Experiment started at: {datetime.datetime.now()}")
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
    config_local = os.path.join(base_folder, config_local)
    config_recon["local"] = config_local
    if config_name:
        config_local_full = os.path.join(config_local, config_name)
        config_recon["file_path"] = config_local_full

    return {**config, **config_recon}


def load_data(data_path):
    if data_path.endswith(".parquet"):
        dataframe = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Input path file format is not supported: {data_path}")

    return dataframe


def isoencode(obj):
    """
    isoencode decodes many different objects such as
    np.bool -> regular bool

    :param obj: input objects to be encoded
    :return: isoencoded object
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)


class DataSet:
    """
    DataSet for the model
    """

    def __init__(self, config, base_folder):
        self.config = config
        self.base_folder = base_folder

        self.pred_cols = self.config.get("targets")
        self.feature_cols = self.config.get("features")
        self.cat_cols = self.config.get("encoding", {}).get("categorical_columns")

        self.test_size = self.config.get("test_size")
        self.random_state = self.config.get("random_state")

        self.artifacts = self.config["artifacts"]

        logger.debug(
            f"features: {self.feature_cols}\n"
            f"predict: {self.pred_cols}\n"
            f"base folder: {self.base_folder}\n"
            f"artifacts configs: {self.artifacts}"
        )

    def _encode(self):
        self.cat_encoder = ce.BinaryEncoder(cols=self.cat_cols)
        self.cat_encoder.fit(self.X, self.y)

    def create_train_test_datasets(self, data):

        self.data = data
        logger.debug(f"length of dataset: {len(self.data)}\n")

        self.X = self.data.loc[:, self.feature_cols]
        self.y = self.data.loc[:, self.pred_cols]

        # No information leak here as we use Binary Encoder only
        self._encode()
        self.X = self.cat_encoder.transform(self.X, self.y)

        logger.debug("Splitting dataset")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        logger.debug(
            "Shape of train and test data:\n"
            f"X_train: {self.X_train.shape}\n"
            f"y_train: {self.y_train.shape}\n"
            f"X_test: {self.X_test.shape}\n"
            f"y_test: {self.y_test.shape}\n"
        )

        # save the train test data
        self._export_train_test_data()

    def _export_train_test_data(self):
        """
        _export_train_test_data saves train and test datasets
        """

        dataset_folder = self.artifacts["dataset"]["local"]

        ## Save the train and test datasets
        logger.info("Export test and train data")
        self.X_train.to_parquet(
            os.path.join(self.base_folder, dataset_folder, "model_X_train.parquet")
        )

        self.X_test.to_parquet(
            os.path.join(self.base_folder, dataset_folder, "model_X_test.parquet")
        )

        pd.DataFrame(self.y_train, columns=self.pred_cols).to_parquet(
            os.path.join(self.base_folder, dataset_folder, "model_y_train.parquet")
        )

        pd.DataFrame(self.y_test, columns=self.pred_cols).to_parquet(
            os.path.join(self.base_folder, dataset_folder, "model_y_test.parquet")
        )

        # Save dataset locally
        self.data.to_parquet(
            os.path.join(self.base_folder, dataset_folder, "dataset.parquet")
        )


class ModelSet:
    """
    The core of the model including hyperparameters
    """

    def __init__(self, config, base_folder):
        self.config = config
        self.base_folder = base_folder

        self.pred_cols = self.config.get("targets")
        self.feature_cols = self.config.get("features")

        self.test_size = self.config.get("test_size")
        self.random_state = self.config.get("random_state")

        self.artifacts = self.config["artifacts"]

        logger.debug(
            f"features: {self.feature_cols}\n"
            f"predict: {self.pred_cols}\n"
            f"base folder: {self.base_folder}\n"
            f"artifacts configs: {self.artifacts}"
        )

    def create_model(self):

        logger.info("Setting up hyperparameters ...")
        self._combine_hyperparameters()

        logger.info("Create pipeline")
        rf = RandomForestRegressor(
            random_state=self.random_state, oob_score=False, n_jobs=-1
        )

        self.pipeline_steps = [("model", rf)]
        self.pipeline = Pipeline(self.pipeline_steps)

        logger.info("Create model with CV")
        self.model = RandomizedSearchCV(
            self.pipeline,
            cv=self.config.get("cv", {}).get("folds", 3),
            n_iter=self.config.get("cv", {}).get("n_iter", 5),
            param_distributions=self.hyperparams_grid,
            verbose=6,
        )

    def _combine_hyperparameters(self):
        hyperparams_grid = self.config.get("hyperparameters")
        if hyperparams_grid is None:
            hyperparams_grid = self._create_hyperparameter_space()
        else:
            hyperparams_grid = {
                **(self._create_hyperparameter_space()),
                **hyperparams_grid,
            }
            logger.info(f"Using hyperparameters: \n{hyperparams_grid}")
        self.hyperparams_grid = hyperparams_grid

    @staticmethod
    def _create_hyperparameter_space():
        """
        _create_hyperparameter_space creates a set of hyperparameters for the random forest
        """

        # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(50, 150, 5)]
        n_estimators = [90, 100, 110, 120]
        # Number of features to consider at every split
        max_features = ["auto", 0.9, 0.8]
        # Maximum number of levels in tree
        # max_depth = [int(x) for x in range(10, 20, 2)]
        max_depth = [None]
        # max_depth.append(None)
        # Minimum number of samples required to split a node
        # min_samples_split = [2, 4, 6]
        min_samples_split = [2]
        # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 3]
        min_samples_leaf = [1]
        # Method of selecting samples for training each tree
        bootstrap = [True]

        # feature_selection__k = [15, 20, 25, 30, 35, 40, 45, 50]

        rf_random_grid = {
            "model__n_estimators": n_estimators,
            "model__max_features": max_features,
            "model__max_depth": max_depth,
            "model__min_samples_split": min_samples_split,
            "model__min_samples_leaf": min_samples_leaf,
            "model__bootstrap": bootstrap,
        }

        return rf_random_grid


class RandomForest:
    """A model to predict the duration"""

    def __init__(self, config, dataset, modelset, base_folder=os.getenv("BASE_FOLDER")):
        self.name = "marshall random forest"
        self.config = config
        self.base_folder = base_folder
        self.artifacts = self.config["artifacts"]
        self.DataSet = dataset
        self.ModelSet = modelset

    def fit_and_report(self):
        """
        _fit_and_report fits the model using input data and generate reports
        """

        logger.info("Fitting the model")
        logger.debug(
            "Shape of train data:\n"
            f"X_train: {self.DataSet.X_train.shape}, {self.DataSet.X_train.sample(3)}\n"
            f"y_train: {self.DataSet.y_train.shape}, {self.DataSet.y_train.sample(3)}"
        )
        self.ModelSet.model.fit(
            self.DataSet.X_train.squeeze(), self.DataSet.y_train.squeeze()
        )

        logger.info("Reporting results")
        train_score = self.ModelSet.model.score(
            self.DataSet.X_train, self.DataSet.y_train
        )
        test_score = self.ModelSet.model.score(self.DataSet.X_test, self.DataSet.y_test)

        self.report = {
            "hyperparameters": self.ModelSet.hyperparams_grid,
            "best_params": self.ModelSet.model.best_params_,
            "cv_results": self.ModelSet.model.cv_results_,
            "test_score": test_score,
            "train_score": train_score,
        }

        logger.debug(self.report)

    def export_results(self):
        """
        export_results saves the necessary artifacts
        """
        model_folder = self.artifacts["model"]["local"]
        if not os.path.exists(os.path.join(self.base_folder, model_folder)):
            os.makedirs(os.path.join(self.base_folder, model_folder))

        logger.info("Preserve models")
        joblib.dump(
            self.ModelSet.model,
            os.path.join(
                self.base_folder,
                model_folder,
                self.artifacts["model"]["name"],
            ),
        )

        logger.info("Perserve logs")
        log_file_path = os.path.join(
            self.base_folder,
            model_folder,
            f'{self.artifacts["model"]["name"]}.log',
        )
        logger.info(f"Save log file to {log_file_path}")
        with open(log_file_path, "a+") as fp:
            json.dump(self.report, fp, default=isoencode)
            fp.write("\n")
        logger.info(f"Saved logs")

    def train(self, dataset):
        """
        train connects the training workflow
        """

        logger.info("1. Create train test datasets")
        self.DataSet.create_train_test_datasets(dataset)
        logger.info("2. Create model")
        self.ModelSet.create_model()
        logger.info("3. Fit model and report")
        self.fit_and_report()
        logger.info("4. Export results")
        self.export_results()


class Reload:
    def __init__(self):
        pass

    def reload(self):
        """
        reload take the saved model and load it back
        """
        model_folder = self.artifacts["model"]["local"]
        if not os.path.exists(os.path.join(self.base_folder, model_folder)):
            raise Exception(
                f"Model folder {os.path.join(self.base_folder, model_folder)} does not exist"
            )

        logger.info("Reload models")
        self.model = joblib.load(
            os.path.join(
                self.base_folder,
                model_folder,
                self.artifacts["model"]["name"],
            )
        )

    def predict(self, dataset):

        res = None

        # reload the model from artifacts
        self.reload()

        # preprocess the data
        # not needed

        # predict
        res = self.model.predict(dataset[self.feature_cols].squeeze())

        return res


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=os.getenv("CONFIG_FILE"),
    help="Path to config file",
)
def preprocess(config):

    base_folder = os.getenv("BASE_FOLDER")

    _CONFIG = load_config(config)
    preprocessed_data_config = get_config(
        _CONFIG, ["preprocessing", "dataset", "preprocessed"]
    )
    preprocessed_data_config = construct_paths(preprocessed_data_config, base_folder)
    rf_config = get_config(_CONFIG, ["model", "rf"])

    # create folders
    if not os.path.exists(preprocessed_data_config["local"]):
        os.makedirs(preprocessed_data_config["local"])

    # load transformed data
    logger.debug(f"Loading data ...")
    df = load_data(preprocessed_data_config["file_path"]).sample(1000)

    # model
    logger.debug(f"Prepare modelset and dataset")
    D = DataSet(config=rf_config, base_folder=base_folder)
    M = ModelSet(config=rf_config, base_folder=base_folder)

    logger.debug(f"Assemble randomforest")
    rf = RandomForest(config=rf_config, dataset=D, modelset=M)

    logger.debug(f"Training ...")
    rf.train(df)


if __name__ == "__main__":
    preprocess()
    print("END OF GAME")
