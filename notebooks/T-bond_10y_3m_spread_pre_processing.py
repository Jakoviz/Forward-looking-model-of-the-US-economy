# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# %%
# Settings:
pd.set_option('display.width', 190)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 70)
pd.set_option('max_colwidth', 200)
pd.options.display.float_format = '{:.4f}'.format
plt.style.use('default')
np.set_printoptions(threshold = 30, edgeitems = 30, precision = 2, suppress = False)

# %%
df = pd.read_csv("../original_data/yield curve data.csv")
df = df.iloc[:, [0,4]]
df.columns = ["Date", "Tbond_10Y3M_spread"]
df.Date = pd.to_datetime(df.iloc[:,0], format='%Y-%m-%d')
df.Tbond_10Y3M_spread = df.Tbond_10Y3M_spread.astype(float)
df.filter(items=['Tbond_10Y3M_spread']).hist(column = 'Tbond_10Y3M_spread', bins = 50) # Histogram before log transformation
df = df.dropna()
df = df.set_index("Date")
df = df.resample(rule="d").first().interpolate()
# %%
# difference transformation
# df = df.diff()
# df = df.drop(index='1967-02-09')

# %%
target = pd.read_csv("../original_data/USRECD.csv", parse_dates=["DATE"])
target = target.rename({"DATE": "Date"}, axis=1)
# target.Date = pd.to_datetime(target.Date)
df = pd.merge(target, df, on="Date")
# %%
df = df.set_index("Date")
df = df.dropna()
# %%
# Histograms of features
features = ['Tbond_10Y3M_spread']

def plot():
    for feature in features:
        df.hist(column = feature, bins = 50)
        plt.xlabel(feature,fontsize=15)
        plt.ylabel("Frequency",fontsize=15)
        plt.show()
        df.plot(y=feature)

# plot()

# %%
df.to_csv("../merged_data/Tbond_10Y3M_spread.csv")



# %%
