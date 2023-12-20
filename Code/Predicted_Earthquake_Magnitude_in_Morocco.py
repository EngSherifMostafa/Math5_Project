import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Reads the dataset from a CSV file into a pandas DataFrame (2D array)
data = pd.read_csv('morocco.csv')

# Extract latitude, longitude, and magnitude columns from the data
latitude = data['Latitude']
longitude = data['Longitude']
magnitude = data['Magnitude']

# Combine latitude & longitude in numpy object
# numpy array holds data from diff types and size
# magnitude into a single feature matrix
X = np.column_stack((latitude, longitude))
y = data['Magnitude']

# Plot the latitude against magnitude
plt.bar(latitude, magnitude)
plt.title('Location vs. Magnitude')
plt.xlabel('Lat, Long')
plt.ylabel('Magnitude')
plt.show()

# split using four args (feature matrix, target vector, test size, random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# create instance from class RandomForestRegressor()
model = RandomForestRegressor()
# training model on the training set using (feature matrix) & (target vector) args
model.fit(X_train, y_train)
# make prediction on the test set (20%) using instance "model"
predictions = model.predict(X_test)

# measure average square difference using target vector as actual values & predictions values from model
mse = mean_squared_error(y_test, predictions)
print("Mean squared error:", round(mse, 5))

Lat = float(input('Latitude: '))
long = float(input('Longitude: '))

predicted_earthquake_magnitude = model.predict([[Lat, long]])
print('Predicted Earthquake Magnitude: ', predicted_earthquake_magnitude)
