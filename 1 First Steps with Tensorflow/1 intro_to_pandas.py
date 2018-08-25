# Basic Concepts

from __future__ import print_function

import pandas as pd
pd.__version__

import numpy as np
np.__version__

# DataFrame - Rows and Columns
# Series - Single Column

# Creating a Series is to construct a Series Object
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

# Creating a Series using a dict mapping String Column
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([872389, 1038237, 3124244])

pd.DataFrame({'City Name': city_names, 'Population': population})

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

# Graphing using Pandas
california_housing_dataframe.hist('housing_median_age')

# Accessing Data
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))
cities['City name']

print(type(cities['City name'][1]))
cities['City name'][1]

print(type(cities[0:2]))
cities[0:2]

# Manipulating Data
population/1000

"""
NumPy is a popular toolkit for scientific computing. 
pandas Series can be used as arguments to most NumPy functions.
"""

np.log(population)

# Below function indicates whether, population
# is over 1 million or not

population.apply(lambda val: val>1000000)

# Adding two Series into a DataFrame
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities

# Excersise 1
# City Size > 50 miles
cities['Area Greater than 50 sq. mile'] = cities['Area square miles'].apply(lambda val: val > 50)

# City name starts with San
# San means Saint in Spanish
cities['Saint Named'] = cities['City name'].apply(lambda city_name: city_name.startswith('San'))


# Doing both at the same time
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
cities

# Indexes
"""
Both Series and DataFrame objects also
define an index property that assigns an identifies value to 
each Series item or DataFrame row
"""

city_names.index
cities.index

# Call DataFrame.reindex to manually
# reorder the rows

cities.reindex([2,0,1])






