import math
from IPython import display
from matplotlib import (
        cm,
        gridspec,
        pyplot as plt,
)
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Loading the Dataset
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Randomize and Scaling the data
california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

# Examine the Data
california_housing_dataframe.describe()

# Build the first model
# In this exercise we will try to predict median_house_value which
# will be our label (sometimes also caled target)

# We will use total_rooms as our input feature

# Define the input feature : total_rooms
my_feature = california_housing_dataframe[['total_rooms']]

# Configure a numeric feature column for total rooms
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# Define the Label
targets = california_housing_dataframe['median_house_value']


# Configure the LinearRegressor

# Use gradient descent as the optimizer for training the model
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = feature_columns,
        optimizer = my_optimizer
)

# Define the input function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trains a linear regression model of one feature.
    Args:
        features:  pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of Batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data or not
        num_epochs: Number of Epochs for which data should be repeated or not
    Returns:
        Tuple of (features, labels) for next data batch
    """
    
    # Convert Pandas data into a dict of np Arrays
    features = {key: np.array(value) for key, value in dict(features).items()}
    
    # Construct a Dataset
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data if Specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    
    # Return the nesxt batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Train the Model
model = linear_regressor.train(
    input_fn = lambda: my_input_fn(my_feature, targets),        
    steps = 100,
)

# Evaluation of the Model
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as Numpy Array, so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])


# Print Mean Squared Error and Root Mean Squared Error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squraed_error = math.sqrt(mean_squared_error)
print("Mean Squared Error : {}".format(mean_squared_error))
print("Root Mean Squared : {}".format(root_mean_squraed_error))

"""
Is this a good model? How would you judge how large this error is?

Mean Squared Error (MSE) can be hard to interpret, so 
we often look at Root Mean Squared Error (RMSE) instead. 
A nice property of RMSE is that it can be interpreted on 
the same scale as the original targets.
Let's compare the RMSE to the difference of the min and max of our targets:

"""

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squraed_error)

# Can we do better ? 
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()

# Plotting the Sample Data
sample = california_housing_dataframe.sample(n=300)




