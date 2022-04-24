# %%
import pandas as pd
import pandas_ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# %%
fileopen = open("CCD_Data.csv")
df = pd.read_csv(fileopen)

# %%
print(df)

# %%
print(df.describe())

# %%
# Reindex data using a DatetimeIndex
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
df = df[['Close']]
print(df)

# %%
print(df.info())

# %%

df.ta.ema(close='close', length=10, append=True)


# %%
print(df)

# %%
df = df.iloc[10:]


# %%
# X_train, X_test, y_train, y_test = train_test_split(
#     df[['Close']], df[['EMA_10']], test_size=.2)
X_train = df['Close'][:int(df.shape[0]*0.75)]
X_test = df['Close'][int(df.shape[0]*0.75):]
y_train = df['EMA_10'][:int(df.shape[0]*0.75)]
y_test = df['EMA_10'][int(df.shape[0]*0.75):]
print(y_test)
print(y_train)

# %%
print(X_test)
print(X_train)


# %%

# Create Regression Model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# %%
# Printout relevant metrics
print("Model Coefficients:", model.coef_)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))


# %%
y_pred

# %%
