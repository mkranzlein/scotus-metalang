import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scotus_metalang.diachronic_analysis import authors