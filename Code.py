import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv('train.csv')
features = ['LotArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
X = data[features]
y = data[target]
model = LinearRegression()
model.fit(X, y)
test_data = pd.read_csv('test.csv')
predicted_prices = model.predict(test_data[features])
submission_data = pd.DataFrame({'Id': test_data['Id'], 'PredictedPrice': predicted_prices})
submission_data.to_csv('house_price_predictions.csv', index=False)
