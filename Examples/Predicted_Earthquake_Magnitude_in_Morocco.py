import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Read the data from the CSV file
data = pd.read_csv('morocco.csv')

# Extract latitude, longitude, and magnitude columns from the data
latitude = data['Latitude']
longitude = data['Longitude']
magnitude = data['Magnitude']

# Combine latitude, longitude, and magnitude into a single feature matrix
X = np.column_stack((latitude, longitude))
y = data['Magnitude']

# Plot the latitude against magnitude
plt.bar(latitude, magnitude)
plt.title('Location vs. Magnitude')
plt.xlabel('Lat, Long')
plt.ylabel('Magnitude')
plt.show()

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Create a model
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean squared error:", round(mse, 2) * 100, "%")

Lat = float(input('Latitude: '))
long = float(input('Longitude: '))

predicted_earthquake_magnitude = model.predict([[Lat, long]])
print('Predicted Earthquake Magnitude: ', predicted_earthquake_magnitude)
