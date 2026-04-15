"""
This script contians all the methods which will be applied in order to clean the MD files.
"""

import os
import logging


#configuring logging
logger = logging.basicConfig(
    level= logging.INFO,
    format= '%(astime)s - %(name)s - %(levelname)s - %(message)s ',
    datefmt='%Y -%m -%d %H:%M:%S'
)
