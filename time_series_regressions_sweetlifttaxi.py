# Project Description
# 
# Sweet Lift Taxi has collected historical data on taxi requests at airports. To attract more drivers during peak hours, we need to predict the number of taxi requests for the next hour. Build a model for this prediction.

# The RECM metric on the test set should not exceed 48.

# ## Project instructions.

# 1. Download the data and resample for an hour.
# 2. Analyze the data.
# 3. Train different models with different hyperparameters. The test sample should be 10% of the initial dataset. 4. Test the data using the test sample and provide a conclusion.

# ## Data Description

# The data is stored in the file `taxi.csv`.
# The number of orders is in the `num_orders` column.

#%%
# ## Data Visualization

# Library are imported

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display

# Data is displayed

df = pd.read_csv('taxi.csv', index_col=[0], parse_dates=[0])
display(df)

# Relevant information is displayed

df.info()

# Duplicates and their number are shown

print(df.duplicated().sum())
print(df.isnull().sum())

# Metrics for numerical variables are shown

df.describe()

# **Observations**
# 
# 1. It is observed that the column names are in a simple format, there are no null values ​​or duplicate rows, the order metrics have normal values.
# 2. The only parameter to change is the variable type of the datetime column, set it as an index and sort them ascendingly by time.

#%%
# ## Analysis

# Dataframe adjustment
df.sort_index(inplace=True)
df = df.resample('1h').sum()
display(df)
print(df.index.is_monotonic_increasing)
df.info()

# Taxi ordering trend
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['num_orders'], label='Orders per hour')
plt.title('Taxi ordering trend')
plt.xlabel('Date')
plt.ylabel('Orders Number')
plt.legend()
plt.show()

df_plt = df['2018-03':'2018-04'].resample('1h').sum()

decomposed = seasonal_decompose(df_plt['num_orders'], model='additive')

plt.figure(figsize=(6, 8))
plt.subplot(311)
# To display the chart correctly, specify its
# ax-axis equal to plt.gca() (gca = get current axis)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()

data = df['2018-03':'2018-05'].resample('1D').sum()
data['num_orders'] = data['num_orders'] - data['num_orders'].shift()

data = df['2018-03':'2018-05'].resample('1D').sum()
data['num_orders'] = data['num_orders'] - data['num_orders'].shift()

# Calculating moving mean and standard deviation
data['mean'] = data['num_orders'].rolling(15).mean()
data['std'] = data['num_orders'].rolling(15).std()

# Plot only the desired columns
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['num_orders'], label='num_orders', alpha=0.6)
plt.plot(data.index, data['mean'], label='Rolling Mean (15d)', color='green')
plt.plot(data.index, data['std'], label='Rolling Std (15d)', color='orange')

plt.title('Daily orders and mobile statistics')
plt.xlabel('Date')
plt.ylabel('Number of orders')
plt.legend()
plt.tight_layout()
plt.show()


# We can see from the graphs that it is a stochastic process since the distribution does not depend on time, it has a trend that can be seen when graphing a month and it has a season, this tells us that we can predict with this data.

#%%

# ## Models

# Function to create the features
def make_features(df, max_lag, rolling_mean_size):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    
    for lag in range(1, max_lag+1):
        df['lag_{}'.format(lag)] = df['num_orders'].shift(lag)
        
    df['rolling_mean'] = df['num_orders'].shift().rolling(rolling_mean_size).mean()


# Division into training and testing
make_features(df, 6, 10)

df1 = df.dropna()
X = df1.drop(['num_orders'], axis = 1)
y = df1['num_orders']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size = 0.1, random_state=12345)

# Model 1
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_train = model_lr.predict(X_train)
rmse_lr = np.sqrt(mean_squared_error(y_train, pred_train))

print("\nModel: Linear Regression")
print(f"RECM: {rmse_lr:.2f}")

# XGB Regressor model 4
# Model configuration
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.2,
    'max_depth': 5,
    'n_estimators': 50,
}

xgb_model = XGBRegressor(**params)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_train)

rmse_xgb = np.sqrt(mean_squared_error(y_train, pred_xgb))

print("\nModel: XGBRegressor")
print(f"RECM: {rmse_xgb:.2f}")

# LightGBM model 5 

# Model configuration
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'max_depth': 10,
    'n_estimators': 100,
}

lgb_model = lgb.LGBMRegressor(**params)
lgb_model.fit(X_train, y_train)

# Prediction and evaluation
y_pred_lgbm = lgb_model.predict(X_train)
rmse_lgbm = mean_squared_error(y_train, y_pred_lgbm)
rmse_lgbm = np.sqrt(rmse_lgbm)

print("\nModel: LightGBM")
print(f"RECM: {rmse_lgbm:.2f}")

#%%

# ## Test

# Model 1

pred_test = model_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_test))
scores = cross_val_score(model_lr, X_test, y_test, cv=5)

print(f"RECM: {rmse_lr:.2f}")
print(f"\nModel: Linear Regression)")
print("CV:", scores.mean())

# Desicion Tree Model 2
max_depth_values = [5, 10, 15, 20]

for depth in max_depth_values:
    dt_model = DecisionTreeRegressor(max_depth=depth, random_state=12345)
    dt_model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred_dt = dt_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred_dt)
    rmse = np.sqrt(rmse)
    
    scores = cross_val_score(dt_model, X_test, y_test, cv=5)

    print(f"\nModel: Decision Tree (max_depth={depth})")
    print(f"RECM: {rmse:.2f}")
    print("CV:", scores.mean())

# Random Forest Model 3

# Hyperparameters to tune
n_estimators_values = [50, 75, 100]
max_depth_values=[5, 10, 15]

for estimators in n_estimators_values:
    for max_depth in max_depth_values:
        rf_model = RandomForestRegressor(n_estimators=estimators, max_depth=max_depth, random_state=12345)
        rf_model.fit(X_train, y_train)

        # Prediction and evaluation
        y_pred = rf_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(rmse)
        scores = cross_val_score(rf_model, X_test, y_test, cv=5)

        print(f"\nModel: Random Forest (n_estimators={estimators} y max_depth={max_depth})")
        print(f"RECM: {rmse:.2f}")
        print("CV:", scores.mean())
    del rf_model, rmse, y_pred

# XGB Regressor model 4

# Prediction and evaluation
pred_xgb = xgb_model.predict(X_test)
rmse_xgb = mean_squared_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(rmse_xgb)
scores = cross_val_score(xgb_model, X_test, y_test, cv=5)

print("\nModel: XGBoost")
print(f"RECM: {rmse_xgb:.2f}")
print("CV:", scores.mean())

# LightGBM model 5 
# Prediction and evaluation
y_pred_lgbm = lgb_model.predict(X_test)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(rmse_lgbm)
scores = cross_val_score(lgb_model, X_test, y_test, cv=5)

print("\nModel: LightGBM")
print(f"RECM: {rmse_lgbm:.2f}")
print("CV:", scores.mean())

#%%
# ## Conclusion

# 1. The model that showed the best performance was LightGBM with the lowest error in cross-validation and the lowest error by RECM.
# 2. In general terms, the Random Forest algorithm also showed a good performance, improving the required error value; however, it turns out to be slower than LightGBM
# 3. It can predict to a good extent the days in which there will be more taxi orders