"""
This module can convert all the pdf files into json format.
It can easily separate the content into different sections.
"""

import os
import json
import logging
from pathlib import Path
from Langchain_unstructured import UnstructuredLoader

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FileLoader:
    """
    Load the files from the folder path
    
    def _validate: Validates if the folder exists or not
    def Loads: loads the Langchain
    """
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self._validate()

    def _validate(self):
        if not self.folder_path.exists():
            logger.error(f"Such folder does not exist: {self.folder_path}")
            raise FileNotFoundError(f"Such folder does not exist: {self.folder_path}")

        else:
            logger.info(f"The folder exists: {self.folder_path}")
            return True

    
    def load(self):
        pdf_files = list(self.folder_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.folder_path}")
        return pdf_files


class PDFProcessor:
    """
    This class processes the PDF file and returns the content in the from of JSON

    """
    @staticmethod
    def loader(file_path, mode = "elements", strategy = "fast"):
        try:
            loader = UnstructuredLoader(file_path, mode=mode, strategy = strategy )
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} elements from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise