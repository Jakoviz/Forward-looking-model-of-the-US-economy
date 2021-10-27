# %%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from random import randrange
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score

# %%
# Settings:
pd.set_option('display.width', 190)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
pd.set_option('max_colwidth', 200)
pd.options.display.float_format = '{:.4f}'.format
plt.style.use('default')
np.set_printoptions(threshold = 30, edgeitems = 30, precision = 2, suppress = False)


# %%
df_path = "/home/jaakkpaa/Documents/Source/IntroToDS/merged_data/CAPE_BAAFFM_BCI_CCI_CLI_IJC_PMIC_T10Y2Y_T10Y3M_vs_USRESCD_GDP.csv"
features = ['CAPE', "BAAFFM", "BCI", "CCI", "CLI", "IJC", "PMIC", "T10Y2Y", "T10Y3M"]
model_names_classification = ["Logistic Regression", "Penalized SVM", "Random Forest"]
model_names_regression = ["Linear Regression"]
get_classification_models = [lambda: linear_model.LogisticRegression(), lambda: svm.SVC(kernel='linear',\
    class_weight='balanced', probability=True), lambda: RandomForestClassifier()]
get_regression_models = [lambda: linear_model.LinearRegression()]
target_variables_classification = ["USRECD"]
target_variables_regression = ["GDP_rate", "GDP_abs"]
target_variables = target_variables_classification + target_variables_regression

# %%
# Read the data and do a little bit of wrangling:
df = pd.read_csv(df_path)
df.Date = pd.to_datetime(df.Date)
df = df.set_index("Date", drop=True)
df = df.dropna()

# %%
# Split into training and test sets and hold out the test set until the end, so that it remains "unseen".
lag_of_y = 180 # This is the lag in days we introduce to the predictor variable so that we assess 
              # the indicator's ability to predict the target variable this many steps into the future.

df_x = df.shift(lag_of_y, freq="d")[df.columns.difference(target_variables)]
df_y = df[target_variables]
        
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1, shuffle=False)

# %%
# Histrograms of the features in training set before any scaling
for feature in features:
    plt.figure()
    X_train[feature].hist(bins = 50)
    plt.xlabel(feature,fontsize=15)
    plt.ylabel("Frequency",fontsize=15)
    plt.show()

# %% [markdown]
# ## Validation

# %%
# Do a time series cross-validation on the test set by splitting it into k folds and doing a "rolling"
# validation against a validation fold, then averaging out the metrics.
splits = 3 # This is the number of splits/folds in the rolling validation.
tscv = TimeSeriesSplit(n_splits=splits)
pd.options.mode.chained_assignment = None

AUC_ROCs = dict()
ACCs = dict()

# %%
## Classification
for target_variable in target_variables_classification:
    print(target_variable)
    for model_name, get_model in zip(model_names_classification, get_classification_models):
        print(model_name)
        AUC_ROCs[model_name] = 0
        ACCs[model_name] = 0
        for train_index, test_index in tscv.split(X_train): # Rolling cross-validation happens inside this loop.
            X_train_fold, X_validation_fold = X_train.iloc[train_index,], X_train.iloc[test_index,]
            y_train_fold, y_validation_fold = y_train.iloc[train_index,][[target_variable]], y_train.iloc[test_index,][[target_variable]]
                
            scalers = dict()
            for feature in features:
                scalers[feature] = StandardScaler()
                scalers[feature].fit(X_train_fold[[feature]]) # Do z-score scaling on the training fold...
                X_train_fold[feature] = scalers[feature].transform(X_train_fold[[feature]])
                X_validation_fold[feature] = scalers[feature].transform(X_validation_fold[[feature]]) # ...and use the same
                # scaling parameters to transform the validation fold to the same scale with it (to avoid data leakage)
                
            model = get_model()
            model.fit(X_train_fold[features], y_train_fold[target_variable])
            positive_probs = [p[1] for p in model.predict_proba(X_validation_fold[features])]
            AUC_ROC = metrics.roc_auc_score(y_validation_fold, positive_probs)
            AUC_ROCs[model_name] += AUC_ROC
            predictions = model.predict(X_validation_fold[features])
            ACC = accuracy_score(y_validation_fold, predictions)
            ACCs[model_name] += ACC
            print("AUC_ROC:", AUC_ROC, "\tAccuracy:", ACC)
            
        AUC_ROCs[model_name] /= splits
        ACCs[model_name] /= splits

for model_name in model_names_classification:
    print(model_name)
    print(f"Average training AUC ROC: {AUC_ROCs[model_name]}")
    print(f"Average training accuracy: {ACCs[model_name]}")

# %%
## Regression
R2s = dict()
for target_variable in target_variables_regression:
    for model_name, get_model in zip(model_names_regression, get_regression_models):
        print(model_name)
        R2s[model_name] = 0
        for train_index, test_index in tscv.split(X_train): # Rolling cross-validation happens inside this loop.
            X_train_fold, X_validation_fold = X_train.iloc[train_index,], X_train.iloc[test_index,]
            y_train_fold, y_validation_fold = y_train.iloc[train_index,], y_train.iloc[test_index,]
                
            scalers = dict()
            for feature in features:
                scalers[feature] = StandardScaler()
                scalers[feature].fit(X_train_fold[[feature]])
                X_train_fold[feature] = scalers[feature].transform(X_train_fold[[feature]])
                X_validation_fold[feature] = scalers[feature].transform(X_validation_fold[[feature]])
                
            model = get_model()
            model.fit(X_train_fold[features], y_train_fold[target_variable])
            predictions = model.predict(X_validation_fold[features])
            R2 = r2_score(y_validation_fold[target_variable], predictions)
            R2s[model_name] += R2
            print(R2)
            
        R2s[model_name] /= splits

# %%
for model_name in model_names_regression:
    print(model_name)
    print(f"Average training R2 score: {R2s[model_name]}")

# %% [markdown]
# ## Test

# %%
y_test.value_counts()

# %%
X_train = X_train.copy()
X_test = X_test.copy()

all_scalers = dict()
for feature in features:
    all_scalers[feature] = StandardScaler()
    all_scalers[feature].fit(X_train[[feature]]) # Do z-score scaling on the training set/split...
    X_train[feature] = all_scalers[feature].transform(X_train[[feature]])
    X_test[feature] = all_scalers[feature].transform(X_test[[feature]]) # ...and use the same
    # scaling parameters to transform the test set/split to the same scale with it (to avoid data leakage)

# %%
for feature in features:
    print(all_scalers[feature].mean_)

# %%
## Classification
for target_variable in target_variables_classification:
    for model_name, get_model in zip(model_names_classification, get_classification_models):
        print(model_name)
        model = get_model()
        model.fit(X_train[features], y_train[target_variable])
        positive_probs = [p[1] for p in model.predict_proba(X_test[features])]
        AUC_ROC = metrics.roc_auc_score(y_test[target_variable], positive_probs)
        fpr, tpr, thresholds = metrics.roc_curve(y_test[target_variable], positive_probs)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
        display.plot()
        predictions = model.predict(X_test[features])
        ACC = accuracy_score(y_test[target_variable], predictions)
        print("Test AUC_ROC:", AUC_ROC, "\nTest Accuracy:", ACC)

    total = y_test.shape[0]
    print("Baseline #1: naively always guess 0 (no recession)\nROC AUC:", 
        metrics.roc_auc_score(y_test[target_variable], np.zeros(total)), 
        "\nAccuracy:", accuracy_score(y_test[target_variable], np.zeros(total)))

    nums = np.zeros(total)
    nums[:total//10] = 1
    np.random.shuffle(nums)
    print("Baseline #2: naively guess 1 (recession) random 10% of the time:", 
        metrics.roc_auc_score(y_test[target_variable], nums), 
        "\nAccuracy:", accuracy_score(y_test[target_variable], nums))

    nums = np.zeros(total)
    for i in range(0, total//3650):
        start = randrange(total)
        for j in range(0, 365):
            nums[start + j] = 1
    print("Baseline #3: naively guess 1 (recession) occurs once for every 3650 days, but at a random point in time, and so that there is a recession 365 days onwards from that point in time", 
        metrics.roc_auc_score(y_test[target_variable], nums), 
        "\nAccuracy:", accuracy_score(y_test[target_variable], nums))
    
# %%
## Regression
for target_variable in target_variables_regression:
    print()
    print(target_variable)
    for model_name, get_model in zip(model_names_regression, get_regression_models):
        print()
        print(model_name)
        model = get_model()
        model.fit(X_train[features], y_train[target_variable])
        predictions = model.predict(X_test[features])
        R2 = r2_score(y_test[target_variable], predictions)
        print("Test R2:", R2)

    total = y_test.shape[0]
    print("\nBaseline #1: TODO")
# %%
