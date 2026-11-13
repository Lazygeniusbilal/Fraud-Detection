# This file will unzip data and store in the data folder 

# import dependencies
import os
from zipfile import ZipFile
from pathlib import Path
from src import ZIPFILE_PATH, UNZIPFILE_PATH


def data_extraction(path: Path | str):
    
    """
        This function takes input data_path and unzip the data and store the respective folders / files in the given destination.
    """
    # destination 
    os.makedirs(name=UNZIPFILE_PATH, exist_ok= True)
    with ZipFile(file= path, mode= "r") as zip_file:
        zip_file.extractall(path=UNZIPFILE_PATH)
        
if __name__ == "__main__":
    data_extraction(path=ZIPFILE_PATH)