# Check the versions of libraries
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
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

# Load libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
# Load dataset
epl_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/E0.csv")
# shape
print(epl_1718.shape)
# head
print(epl_1718.head(20))
# names
print(epl_1718.columns)
# descriptions
print(epl_1718.describe())
#epl_1718['Date'] = pd.to_datetime(epl_1718['Date'], format='%d/%m/%y')
#epl_1718['time_diff'] = (max(epl_1718['Date']) - epl_1718['Date']).dt.days
#epl_1718 = epl_1718[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'time_diff']]
epl_1718 = epl_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})

# shape
print(epl_1718.shape)
# head
print(epl_1718.head(20))
# names
print(epl_1718.columns)
# descriptions
print(epl_1718.dtypes)
# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot
sns.countplot('HomeGoals',data=epl_1718,hue = 'HomeTeam')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# display the plot
plt.show()
"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
"""
