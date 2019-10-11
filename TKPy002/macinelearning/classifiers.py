# Check the versions of libraries
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib as plt
print('matplotlib: {}'.format(plt.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# joblib
import joblib as jl
print('joblib: {}'.format(jl.__version__))
# scikit-learn
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))

import sklearn_pandas as skp
print('sklearn_pandas: {}'.format(skp.__version__))
"""
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
print(iris_data[0], iris_data[79], iris_data[100])
print(iris_labels[0], iris_labels[79], iris_labels[100])
"""