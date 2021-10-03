# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
# %%
# Settings:
pd.set_option('display.width', 190)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('max_colwidth', 200)
pd.options.display.float_format = '{:.4f}'.format
plt.style.use('default')
np.set_printoptions(threshold = 30, edgeitems = 30, precision = 2, suppress = False)
# %%
# Read the data and do a little bit of wrangling:
df = pd.read_csv("BCI-values.csv")
target = pd.read_csv("USRECD.csv")
target = target.rename({"DATE": "Date"}, axis=1)
df.Date = pd.to_datetime(df.Date)
target.Date = pd.to_datetime(target.Date)
df = pd.merge(df, target, on="Date")
df = df.sort_values(ascending=True, by="Date")
df = df.set_index("Date")

# %%
# Histograms of features before logarithmic transformation:
df.hist(column = "BCI", bins = 50)
plt.xlabel("BCI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCI")
# %%
df.hist(column = "BCIp", bins = 50)
plt.xlabel("BCIp",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCIp")
# %%
df.hist(column = "BCIg", bins = 50)
plt.xlabel("BCIg",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCIg")

# %%
# Log transformations on the data:
df.BCI = df.BCI.apply(lambda x: np.log(x))
df.BCIg = df.BCIg.apply(lambda x: np.log(x + abs(min(df.BCIg)) + 0.1)) # Because there are values that are negative, we transform all values just a little bit positive to be able to make a logarithmic transformation.
df.BCIp = df.BCIp.apply(lambda x: np.log(x + abs(min(df.BCIp)) + 0.1))

# %%
# Histograms of features after logarithmic transformation:
df.hist(column = "BCI", bins = 50)
plt.xlabel("BCI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCI")
# %%
df.hist(column = "BCIp", bins = 50)
plt.xlabel("BCIp",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCIp")
# %%
df.hist(column = "BCIg", bins = 50)
plt.xlabel("BCIg",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
df.plot(y="BCIg")
# %%
# Split into training (+validation) and test sets
tscv = TimeSeriesSplit()
print(tscv)
for train_index, test_index in tscv.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = df.iloc[train_index, df.columns != "USRECD"], df.iloc[test_index, df.columns != "USRECD"]
    y_train, y_test = df.iloc[train_index, df.columns == "USRECD"], df.iloc[test_index, df.columns == "USRECD"]

# %%
# Standardization/z-score normalization/scaling for training set:
scaler = StandardScaler()
X_train = X_train.copy()
X_train['BCI'] = scaler.fit_transform(X_train['BCI'].values.reshape(-1,1))
z_score_scaler_bci = scaler.fit(X_train['BCI'].values.reshape(-1,1))
X_train['BCIp'] = scaler.fit_transform(X_train['BCIp'].values.reshape(-1,1))
z_score_scaler_bcip = scaler.fit(X_train['BCIp'].values.reshape(-1,1))
X_train['BCIg'] = scaler.fit_transform(X_train['BCIg'].values.reshape(-1,1))
z_score_scaler_bcig = scaler.fit(X_train['BCIg'].values.reshape(-1,1))

# No need to do this on the US recession feature, because it is a binary variable with values 0 and 1. This should be done on the GDP growth rate continuous variable, however.
#
# y_train = y_train.copy()
# y_train['USRECD'] = scaler.fit_transform(y_train['USRECD'].values.reshape(-1,1))
# z_score_scaler_usrecd = scaler.fit(y_train['USRECD'].values.reshape(-1,1))

# %%
# Histograms of training set after z-score normalization:
X_train.BCI.hist(bins = 50)
plt.xlabel("BCI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCI.plot(x="Date", y="BCI")
# %%
X_train.BCIp.hist(bins = 50)
plt.xlabel("BCIp",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCIp.plot(x="Date", y="BCIp")
# %%
X_train.BCIg.hist(bins = 50)
plt.xlabel("BCIg",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCIg.plot(x="Date", y="BCIg")

# %%
# Perhaps z-score normalization works better here than min-max so thus this is commented out.
#
# Min-max normalization/scaling for training set:
# scaler = MinMaxScaler()
# X_train = X_train.copy()
# X_train['BCI'] = scaler.fit_transform(X_train['BCI'].values.reshape(-1,1))
# min_max_scaler_bci = scaler.fit(X_train['BCI'].values.reshape(-1,1))
# X_train['BCIp'] = scaler.fit_transform(X_train['BCIp'].values.reshape(-1,1))
# min_max_scaler_bcip = scaler.fit(X_train['BCIp'].values.reshape(-1,1))
# X_train['BCIg'] = scaler.fit_transform(X_train['BCIg'].values.reshape(-1,1))
# min_max_scaler_bcig = scaler.fit(X_train['BCIg'].values.reshape(-1,1))

# No need to do this on the US recession feature, because it is a binary variable with values 0 and 1. This should be done on the GDP growth rate continuous variable, however.
#
# y_train = y_train.copy()
# y_train['USRECD'] = scaler.fit_transform(y_train['USRECD'].values.reshape(-1,1))
# min_max_scaler_usrecd = scaler.fit(y_train['USRECD'].values.reshape(-1,1))

# %%
# Histograms of training set after min-max normalization:
X_train.BCI.hist(bins = 50)
plt.xlabel("BCI",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCI.plot(x="Date", y="BCI")
# %%
X_train.BCIp.hist(bins = 50)
plt.xlabel("BCIp",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCIp.plot(x="Date", y="BCIp")
# %%
X_train.BCIg.hist(bins = 50)
plt.xlabel("BCIg",fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.show()
X_train.BCIg.plot(x="Date", y="BCIg")

# %%
# # Standardization/z-score normalization/scaling for testing set:
X_test = X_test.copy()
z_score_scaler_bci = z_score_scaler_bci.fit(X_test['BCI'].values.reshape(-1,1))
X_test['BCI'] = z_score_scaler_bci.transform(X_test['BCI'].values.reshape(-1,1))

z_score_scaler_bcip = z_score_scaler_bcip.fit(X_test['BCIp'].values.reshape(-1,1))
X_test['BCIp'] = z_score_scaler_bcip.transform(X_test['BCIp'].values.reshape(-1,1))

z_score_scaler_bcig = z_score_scaler_bcig.fit(X_test['BCIg'].values.reshape(-1,1))
X_test['BCIg'] = z_score_scaler_bcig.transform(X_test['BCIg'].values.reshape(-1,1))

# No need to do this on the US recession feature, because it is a binary variable with values 0 and 1. This should be done on the GDP growth rate continuous variable, however.
#
# y_test = y_test.copy()
# z_score_scaler_usrecd = z_score_scaler_usrecd.fit(y_test['USRECD'].values.reshape(-1,1))
# y_test['USRECD'] = z_score_scaler_usrecd.transform(y_test['USRECD'].values.reshape(-1,1))
# %%
