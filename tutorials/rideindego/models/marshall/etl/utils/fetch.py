import os
import re
import zipfile
import time

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .web import random_user_agent as _random_user_agent


def get_page_html(link, retry_params=None, headers=None, timeout=None, session=None):
    """Download html page and save content"""

    if retry_params is None:
        retry_params = {}

    retry_params = {
        **{"retries": 5, "backoff_factor": 0.3, "status_forcelist": (500, 502, 504)},
        **retry_params,
    }

    if headers is None:
        headers = _random_user_agent()

    if timeout is None:
        timeout = (5, 14)

    if session is None:
        session = requests.Session()

    retry = Retry(
        total=retry_params.get("retries"),
        read=retry_params.get("retries"),
        connect=retry_params.get("retries"),
        backoff_factor=retry_params.get("backoff_factor"),
        status_forcelist=retry_params.get("status_forcelist"),
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    page = session.get(link, headers=headers)

    status = page.status_code

    return {"status": status, "data": page}


class DataLinkHTMLExtractor:
    """Extract links to data files from Rideindego html data"""

    def __init__(self, page):
        self.page = page

    @staticmethod
    def _get_zip_links(text_content):
        """Extract zip file links from text"""

        return re.findall("(?P<url>https?://[^\s]+.zip)", text_content)

    @staticmethod
    def _get_csv_links(text_content):
        """Extract csv links from text"""

        return re.findall('(?P<url>https?://[^\s]+.csv)"', text_content)

    def get_data_links(self):

        page = self.page
        page_text = page.text

        # Extract quarter data
        quarterly_zips = self._get_zip_links(page_text)

        if len(quarterly_zips) == 0:
            raise Exception("Did not extract zip file links")
        else:
            return quarterly_zips


class DataDownloader:
    """Download data from links"""

    def __init__(self, links, data_type, folder, filenames=None):
        """
        :param data_type: Data type of the links, can be 'zip' or 'data'
        """
        self.links = links
        self.data_type = data_type
        self.folder = folder
        self.__check_folder_exist()
        if filenames:
            if len(filenames) != len(links):
                raise Exception("len(filenames) do not match len(links)!")
            else:
                self.filenames = filenames

    def __check_folder_exist(self, create=True):
        """Check if a folder is already there; Create folder if does not exist"""

        try:
            os.mkdir(self.folder)
            logger.info("Created folder: ", self.folder)
        except FileExistsError:
            logger.info("Already exists: ", self.folder)

    def _download_binary_files(self):
        """Download binary files"""

        self._binary_files = []
        for idx, link in enumerate(self.links):
            target_file = os.path.join(self.folder, str(idx) + ".zip")
            # Check status: sometimes it fails to download
            # the actual data file thus unzip fails.
            # check the status code and retry
            # will solve this problem
            time.sleep(5)
            file_content = get_page_html(link).get("data")
            try:
                with open(target_file, "wb") as fp:
                    fp.write(file_content.content)
                self._binary_files.append(target_file)
                logger.debug(f"Downloaded {target_file}")
            except Exception as ee:
                logger.error(f"Could not download {link}")
                pass
        logger.info("Downloaded {} zip files!".format(len(self._binary_files)))

    def _unzip_data_files(self):
        self._data_files = []
        for data_file in self._binary_files:
            try:
                with zipfile.ZipFile(data_file, "r") as zip_ref:
                    zip_ref.extractall(self.folder)
                self._data_files.append(data_file[:-4])
            except Exception as ee:
                logger.error(f"Could not unzip data: {data_file}")
                pass

    def _summary(self):
        """Validate the number of files downloaded and unzipped"""

        link_count = len(self.links)
        self.data_files = self._binary_files
        if self.data_type == "zip":
            self.data_files = self._data_files
        data_file_count = len(self.data_files)

        if data_file_count != link_count:
            logger.error(
                f"Data files Errors: links: {link_count}, zips: {data_file_count}"
            )

        res = {
            "folder": self.folder,
            "links": self.links,
            "link_count": link_count,
            "data_files": self.data_files,
            "data_count": data_file_count,
        }

        return res

    def run(self):
        """Connect the pipes to extract csv files"""
        self._download_binary_files()
        if self.data_type == "zip":
            self._unzip_data_files()

        return self._summary()


if __name__ == "__main__":

    # _CONFIG = _get_config()
    _CONFIG = {}

    source_link = _CONFIG.get("etl", {}).get("trip_data", {}).get("source")
    page = get_page_html(source_link).get("data", {})
    page_extractor = DataLinkHTMLExtractor(page)
    links = page_extractor.get_data_links()

    dld = DataDownloader(links)
    dld.run()
    _CONFIG.get("datadir")
    print("END OF GAME")
