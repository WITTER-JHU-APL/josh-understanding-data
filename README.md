# Josh's Data/Graphing Tasks

## CURRENTLY NOT FUNCTIONING PROPERLY

### Purpose
The purpose of these tasks was to use imports and data to create a scatter plot and line of best fit for prices and losses of cars.

### The imports
```python
import math
from IPython import display
import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io, estimator
```
We used this data set for our information: https://storage.googleapis.com/ml_universities/cars_dataset/cars_data.csv

### The Process
First we converted missing or invalid data into 0's 
```python
car_data['price'] = pd.to_numeric(car_data['price'], errors='coerce')
car_data['horsepower'] = pd.to_numeric(car_data['horsepower'], errors='coerce')
car_data['peak-rpm'] = pd.to_numeric(car_data['peak-rpm'], errors='coerce')
car_data['city-mpg'] = pd.to_numeric(car_data['city-mpg'], errors='coerce')
car_data['highway-mpg'] = pd.to_numeric(car_data['highway-mpg'], errors='coerce')
car_data['losses'] = pd.to_numeric(car_data['losses'], errors='coerce')
car_data.fillna(0, inplace=True)
```
Then proceeded to create a graph and RMSE:
```python
def make_scatter_plot(dataframe, input_feature, target,
                      slopes=[], biases=[], model_names=[]):
  """ Creates a scatter plot of input_feature vs target along with the models.
  
  Args:
    dataframe: the dataframe to visualize
    input_feature: the input feature to be used for the x-axis
    target: the target to be used for the y-axis
    slopes: list of model weight (slope) 
    bias: list of model bias (same size as slopes)
    model_names: list of model_names to use for legend (same size as slopes)
  """      
  # Define some colors to use that go from blue towards red
  colors = [cm.coolwarm(x) for x in np.linspace(0, 1, len(slopes))]
  
  # Generate the Scatter plot
  x = dataframe[input_feature]
  y = dataframe[target]
  plt.ylabel(target)
  plt.xlabel(input_feature)
  plt.scatter(x, y, color='black', label="")

  # Add the lines corresponding to the provided models
  for i in range (0, len(slopes)):
    y_0 = slopes[i] * x.min() + biases[i]
    y_1 = slopes[i] * x.max() + biases[i]
    plt.plot([x.min(), x.max()], [y_0, y_1],
             label=model_names[i], color=colors[i])
  if (len(model_names) > 0):
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```
```python
x = car_data[INPUT_FEATURE]
y = car_data[LABEL]
opt = np.polyfit(x, y, 1)
y_pred = opt[0] * x + opt[1]
opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))
slope = opt[0]
bias = opt[1]
print("Optimal RMSE =", opt_rmse, "for solution", opt)
```
I then combined these 2 methods into just the RMSE and make_scatter_plot calls
```python
INPUT_FEATURE = "price"
LABEL = "losses"

# Fill in the rest of this block.
x = car_data[INPUT_FEATURE]
y = car_data[LABEL]
opt = np.polyfit(x, y, 1)
y_pred = opt[0] * x + opt[1]
opt_rmse = math.sqrt(metrics.mean_squared_error(y_pred, y))
slope = opt[0]
bias = opt[1]
print("Optimal RMSE =", opt_rmse, "for solution", opt)
print(opt[0])
print(bias)

make_scatter_plot(car_data,INPUT_FEATURE, LABEL,
                  [slope], [bias], ["initial model"])
```
This final call fully printed and created a scatter plot based on the prices and losses of the cars in the data given.