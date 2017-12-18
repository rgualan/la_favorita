import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics
import gc; gc.enable()
import random
import time, datetime

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor, BayesianRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


np.random.seed(1122)


# Util functions
def print_runtime(start, end):
    print("runtime: {}".format( datetime.timedelta(seconds=(end-start)/60)))

def print_dataframe_size(name, df):
    print("size of {}: {:.3f} MB".format(name, df.memory_usage(index=True).sum()/1E6))


# Read datasets
print('Reading datasets...')
start = time.time()

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

print('Reading train and test...')
train = pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date'])
test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])
print('Reading others...')
items = pd.read_csv('../input/items.csv', dtype={'item_nbr':'int32', 'perishable':bool})
stores = pd.read_csv('../input/stores.csv', dtype={'store_nbr':'uint8', 'cluster':'uint8' })
transactions = pd.read_csv('../input/transactions.csv', dtype={'store_nbr':'uint8'}, parse_dates=['date'])
holidays = pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date'])
holidays['transferred'] = holidays['transferred'].map({'False': 0, 'True': 1})
oil = pd.read_csv('../input/oil.csv', parse_dates=['date'])

print_runtime(start, time.time())


# Dataset processing
print('\nDatasets processing...'); 
start_dp = time.time()

## Reduce training dataset
train = train[(train['date'].dt.month == 8) & (train['date'].dt.day > 15)]
#train = train[(train['date'].dt.month == 8)]
train['onpromotion'] = train['onpromotion'].map({'False': 0, 'True': 1})
test['onpromotion'] = test['onpromotion'].map({'False': 0, 'True': 1})

#train2017 = train[(train['date'].dt.year == 2017)]
#train201608 = train[(train['date'].dt.year == 2016) & (train['date'].dt.day > 15)]
#train2016 = train[(train['date'].dt.year == 2016))]
#train = pd.concat([train2017,train2016])
#del train2017, train2016; gc.collect();

#train[(train['date'].dt.year == 2016) | (train['date'].dt.year == 2017)]
#train = train[(train['date'].dt.year == 2017)]

#train = train[(train['date'].dt.month >= 5)]

# Transform target
target = train['unit_sales'].values
target[target < 0.] = 0.
train['unit_sales'] = np.log1p(target)
# Transforma transactions
#tx = transactions['transactions'].values
#transactions[tx < 0.] = 0.
transactions['transactions'] = np.log1p(transactions['transactions'])


def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

def df_lbl_enc_2(df, cols):
    for c in cols:
        lbl = preprocessing.LabelEncoder()
        df[c] = lbl.fit_transform(df[c])
    return df

def df_transform_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek # Replace!!!
    df = df.fillna(-1) #TODO: WTF???
    return df

# Encode categorical variables
# a)
# items = df_lbl_enc_2(items, ['family'])
# stores = df_lbl_enc_2(stores, ['type','city','state'])

# b)
items = pd.get_dummies(items, columns = ['family'] )
stores = pd.get_dummies(stores, columns = ['type','city','state'] ) 

# Weight for calculating the error metric
items['perishable_w'] = items['perishable'].map({False:1.0, True:1.25}) 

# Create national holidays field
holidays['national_holiday'] = False 
holidays.loc[
    lambda df: (df.type=='Holiday') & (df.locale=='National') & ~(df.transferred), 'national_holiday'
    ] = True
holidays.loc[
    lambda df: (df.type=='Transfer') & (df.locale=='National'), 'national_holiday'
    ] = True


# Merge dataframes
train = pd.merge(train, items, how='left', on=['item_nbr'])
train = pd.merge(train, transactions, how='left', on=['date','store_nbr'])
train = pd.merge(train, stores, how='left', on=['store_nbr'])
train = pd.merge(train, holidays[['date','national_holiday']], how='left', on=['date'])
train = pd.merge(train, oil, how='left', on=['date'])
train = df_transform_date(train)

test = pd.merge(test, items, how='left', on=['item_nbr'])
test = pd.merge(test, transactions, how='left', on=['date','store_nbr'])
test = pd.merge(test, stores, how='left', on=['store_nbr'])
test = pd.merge(test, holidays[['date','national_holiday']], how='left', on=['date'])
test = pd.merge(test, oil, how='left', on=['date'])
test = df_transform_date(test)

del items, transactions, stores, holidays, oil; gc.collect();
print_dataframe_size("train", train)
print_dataframe_size("test", test)
print_runtime(start_dp, time.time())                                                                                                                                                                                                                                                                       


### Predict future transactions 

# Error metric
def NWRMSLE(y, pred, w):
    #return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5
    return metrics.mean_squared_error(y, pred)**0.5

col = [c for c in train if c not in ['id', 'unit_sales','perishable_w','perishable','transactions']]
print(col)

x1 = train[(train['yea'] != 2016)]
x2 = train[(train['yea'] == 2016)]
#x1 = train[(train['mon'] != 8)]
#x2 = train[(train['mon'] == 8)]
x1 = train[ (train['date'] < '2017-08-01') & (train['date'] >= '2016-01-01') ]
x2 = train[(train['date'] >= '2017-08-01')]

y1 = x1['transactions'].values
y2 = x2['transactions'].values
del train; gc.collect();

# debug
#x1.to_csv('../output/x1_v2.csv')
#x2.to_csv('../output/x2_v2.csv')
#raise Exception('debug')



# Forecast
print('\nRunning regressor...')

# set the seed to generate random numbers
method = 1
np.random.seed(round(method + 123*method + 456*method) )

# Model
print('Multilayer perceptron (MLP) neural network 01')
r = MLPRegressor(hidden_layer_sizes=(3,), max_iter=1000)
r.fit(x1[col], y1)
m1 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable_w'])
#tdf = pd.DataFrame( {'y_tx':y2, 'yh_tx':r.predict(x2[col])} )
#tdf.to_csv('../output/pred_tx.csv')
print("Error: %f"%(m1))

test['transactions'] = r.predict(test[col])
test['transactions'] = test['transactions'].clip(lower=0.+1e-12)

#test['transactions'] = (np.expm1(test['transactions'])).clip(lower=cut)



raise Exception('debug!')



### Predict future unit sales 
col = [c for c in x1 if c not in ['id', 'unit_sales','perishable_w','perishable']]
y1 = x1['unit_sales'].values
y2 = x2['unit_sales'].values

# set a new seed to generate random numbers
np.random.seed(round(method + 987*method + 654*method) )

r = MLPRegressor(hidden_layer_sizes=(3,), max_iter=30)
r.fit(x1[col], y1)
m2 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable_w'])
tdf = pd.DataFrame( {'y_us':y2, 'yh_us':r.predict(x2[col])} )
tdf.to_csv('../output/pred_us.csv')

print('Performance: NWRMSLE(1) = ', m1, 'NWRMSLE(2) = ', m2)

test['unit_sales'] = r.predict(test[col])
cut = 0.+1e-12 # 0.+1e-15
test['unit_sales'] = (np.expm1(test['unit_sales'])).clip(lower=cut) # adopted in https://www.kaggle.com/the1owl/forecasting-favorites , version 10

# Save output
test[['id','unit_sales']].to_csv("../output/mlp1.csv", index=False, float_format='%.2f')

print( "\nFinished ...")
print_runtime(start, time.time())
