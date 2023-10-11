#####################################################
# Store Item Demand Forecasting Challenge
#####################################################
## Problem Statement
# Predict 3 months of item sales at different stores

## Dataset Overview
# The dataset contains 5 years of store-item sales data, for 50 different items at 10 different stores.
# The dataset is from a Kaggle competition, and it's divided into three separate CSV files: train, test and sampla_submission.
# The train dataset includes the sales, while the test dataset has the sale prices left blank requiring to predict.
# https://www.kaggle.com/c/demand-forecasting-kernels-only

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


##########################################
# TASK 1 - EXPLORATORY DATA ANALYSIS (EDA)
##########################################

# Read and Combine Train and Test Datasets
train = pd.read_csv('Datasets/demand forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('Datasets/demand forecasting/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('Datasets/demand forecasting/sample_submission.csv')

df = pd.concat([train, test], sort=False)

# Check DataFrame Information
#############################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Other Important Information --> These shows the performance of the stores.
###############################
# Min and Max Dates
df["date"].min(), df["date"].max()

# Number of Items in Stores
df.groupby(["store"])["item"].nunique()

# Sum of Sales by Item and Store
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# Sales Statistics by Store and Item
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# TASK 2 - FEATURE ENGINEERING
#####################################################

# Creating New Date Features
##############################
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

# Define a function to identify Random Noise
########################################
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


# Define a function to add Lag/Shifted Features and Random Noise
#################################################################
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df) # --> These features are oriented in 3 months period because prediction period is 3 months.

# Define a function to identify Rolling Mean Features
#####################################################
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])

# Define a function to identify Exponentially Weighted Mean Features
#####################################################################
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)

# One-Hot Encoding
########################
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)

# Converting sales to log(1+sales) --> The purpose of "log" is to standardize the dependent variable and shorten the train time
###################################
df['sales'] = np.log1p(df["sales"].values) #--> "1" is a method ussed to prevent some possible errors.

check_df(df)

#######################################
# TASK 3 - MODELLING
#######################################

# Custom Cost Function #--> In the competition submissions are evaluated on SMAPE between forecasts and actual values.
########################
#--> MAPE: mean absolute percentage error
#--> SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# Time-Based Validation Sets
#############################

# Train set until the beginning of 2017
train = df.loc[(df["date"] < "2017-01-01"), :]

# Validation set for the first 3 months of 2017
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


# Time Series Model with LightGBM
#####################################

# LightGBM parameters
lgb_params = {'num_leaves': 10, # -->max number of leaves on a tree
              'learning_rate': 0.02,# --> shrinkage_rate
              'feature_fraction': 0.8, # --> same as Random Subspace inf RF. Random number of variables to consider in each iteration
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping': 200, #--> if the error does not decrease, stop modeling.
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping=lgb_params['early_stopping'],
                  feval=lgbm_smape,
                  verbose=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


# İmportance of Variables
######################################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


# Final Model
########################
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand_store.csv", index=False)



