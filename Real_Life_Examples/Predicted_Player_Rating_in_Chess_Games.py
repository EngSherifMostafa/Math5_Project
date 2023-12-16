import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your CSV file
df = pd.read_csv('chess_games.csv')

# Preprocess the data
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
df['ranking_date'] = pd.to_datetime(df['ranking_date'], errors='coerce')  # Use 'coerce' to handle invalid dates
df.sort_values(by='ranking_date', inplace=True)

# Remove rows with missing or invalid dates
df = df.dropna(subset=['ranking_date'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['name', 'previous_rating']], df['rating'], test_size=0.2,
                                                    random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Now, let the user input a player name and future date
user_input_name = input('Enter the player name: ')
user_input_future_date_str = input('Enter the future date (yyyy-mm-dd): ')

# Transform the input using LabelEncoder
user_input_name_encoded = le.transform([user_input_name])[0]

# Convert the user input date to datetime
user_input_future_date = pd.to_datetime(user_input_future_date_str, errors='coerce')

# Find the most recent rating for the given player before or on the future date
recent_rating = df[(df['name'] == user_input_name_encoded) & (df['ranking_date'] <= user_input_future_date)].iloc[-1][
    'rating']

# Make a prediction for the user input
user_input_data = pd.DataFrame([[user_input_name_encoded, recent_rating]], columns=['name', 'previous_rating'])
predicted_rating = model.predict(user_input_data)[0]

print(f'Predicted rating for {user_input_name} on {user_input_future_date}: {predicted_rating}')
