import pandas as pd
from sklearn.linear_model import LinearRegression
# Load training data
data = pd.read_csv('train.csv')
# Define features and target
features = ['LotArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
# Split data into features and target variables
X = data[features]
y = data[target]
# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)
# Load test data
test_data = pd.read_csv('test.csv')
# Predict prices for test data
predicted_prices = model.predict(test_data[features])
# Prepare submission DataFrame (assuming 'Id' is the ID column in test data)
submission_data = pd.DataFrame({'Id': test_data['Id'], 'PredictedPrice': predicted_prices})

# Save the submission DataFrame as CSV (replace with desired filename)
submission_data.to_csv('house_price_predictions.csv', index=False)
