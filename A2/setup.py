#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Standard setup used by all notebooks
"""
# Load external dependencies
import random
import os,glob,re
import numpy as np
import pandas as pd # (*) Pandas for data manipulation
import itertools

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.patheffects as patheffects
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import warnings
warnings.filterwarnings("ignore")

# Load file path for data directory
dir_data = 'data/'
dir_results = 'results/'

