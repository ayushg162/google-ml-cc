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

# Plotting the Data

# Get the min and max total_rooms values
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot out regression line from (x_0, y_0) to (x_1, y_1)
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display Plot
plt.show()


# Tweaking the Model Hyperparameters
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """
    Trains a linear regression model of one feature.
    Args:
        learning_rate: A 'float', the learning rate
        steps: A non-zero 'int', the total number of training steps.
        batch_size: A non-zero 'int', the batch size
        input_feature: A 'string' specifying a column from 'california_dataset',
        to use as input feature
    """
    periods = 10
    steps_per_period = steps / periods
    
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]
    
    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]
    
    # Create input functions
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
    
    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    
    linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer=my_optimizer
    )
    
    # Set up the plotting 
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    # Train the model, but do so inside a loop so that we can period
    # loss metrics
    
    print("Training Model ")
    print("RMSE (on training data) : ")
    
    root_mean_squared_errors = []
    
    for period in range(0, periods):
        linear_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period
        )
        
        # Take a break and compute predictions
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
    
        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])
        
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
        
    print("Model training finished")
     
    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())
    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
    
# Synthetic Features and Outliers
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")
        
        
        
        
        
        
        
        
        
        
        