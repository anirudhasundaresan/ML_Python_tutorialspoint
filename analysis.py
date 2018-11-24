# loading the dataset
import pandas as pd
data = 'pima_indians.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']
dataset = pd.read_csv(data, names=names)

# https://gist.github.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f -- data here

# dimensions of data
print(dataset.shape)

# list entire data
print(dataset.head(20))

# view statistical summaries
print(dataset.describe())

# breakdownn data by class variable
print(dataset.groupby('Outcome').size())

# univariate plots - plots for each individual variable
# box & whisker plots
import matplotlib.pyplot as plt
data = 'iris_df.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(data, names=names)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histogram
dataset.hist() # id the ones with almost gaussian distributions and we can then see which algorithms to apply.
plt.show()

# multivariate plots
# scatter plot matrix - all pairs of attributes
from pandas.plotting import scatter_matrix
scatter_matrix(dataset) # can be used to indicate correlation and diagonal grouping of variables.
plt.show()




