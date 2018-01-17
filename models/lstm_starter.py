"""
This is an upgraded version of Ceshine's and Linzhi and Andy Harless starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
import pickle


# Pre-processing dataset
if False:
    print("Reading files...")

    df_train = pd.read_csv(
        '../input/train.csv', usecols=[1, 2, 3, 4, 5],
        #'../input/processed/train_from2017.csv', usecols=[1, 2, 3, 4, 5],
        dtype={'onpromotion': bool},
        converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
        parse_dates=["date"]
        , skiprows=range(1, 66458909)  # 2016-01-01
    )

    df_test = pd.read_csv(
        "../input/test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

    items = pd.read_csv(
        "../input/items.csv",
    ).set_index("item_nbr")


    #df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
    df_2017 = df_train.loc[df_train.date >= pd.datetime(2016, 1, 1)]
    df_2017 = df_train
    del df_train

    #df_2017 = pd.merge(df_2017, transactions, how='left')

    promo_2017_train = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train

    df_2017 = df_2017.set_index(
        ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
            level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)

    items = items.reindex(df_2017.index.get_level_values(1))

    print("Saving pickles...")
    pickle.dump(df_2017, open('../input/processed/df_2017.pickle', 'wb'))
    pickle.dump(df_test, open('../input/processed/df_test.pickle', 'wb'))
    pickle.dump(promo_2017, open('../input/processed/promo_2017.pickle', 'wb'))
    pickle.dump(items, open('../input/processed/items.pickle', 'wb'))
    # raise Exception("Pickles saved!")

else:
    print("Loading pickles...")
    df_2017 = pickle.load(open('../input/processed/df_2017.pickle', 'rb'))
    promo_2017 = pickle.load(open('../input/processed/promo_2017.pickle', 'rb'))
    items = pickle.load(open('../input/processed/items.pickle', 'rb'))
    df_test = pickle.load(open('../input/processed/df_test.pickle', 'rb'))


stores_items = pd.DataFrame(index=df_2017.index)
test_ids = df_test[['id']]

items = items.reindex(stores_items.index.get_level_values(1))

items_class = pd.get_dummies(items["class"], prefix="class", drop_first=True)
items_class.reset_index(drop=True, inplace=True)

# items_perishable = items["perishable"].reset_index(drop=True)  # Didn't help


# ## Transactions (experimental)
# transactions = pd.read_csv(
#     "../input/processed/sales_completed.csv",
#     parse_dates=["date"]
# )
#
# txs = promo_2017.stack()
# txs = pd.DataFrame(txs)
# txs.reset_index(inplace=True)
# txs = pd.merge(txs, transactions, how='left')
# txs.loc[txs.transactions.isnull(), "transactions"] = 0
# txs = txs.set_index(["store_nbr", "item_nbr", "date"])[["transactions"]].unstack(level=-1).fillna(0)
# txs.columns = txs.columns.get_level_values(1)


def get_timespan(df, dt, minus, periods, freq='D'):
    #return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]
    date_index = [c for c in pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)
                  if c in df.columns]
    return df[date_index]


def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
        #, "mean_tx_14": get_timespan(txs, t2017, 14, 14).mean(axis=1).values
        , "mean_365_2017": get_timespan(df_2017, t2017, 365-8, 16).mean(axis=1).values,  # yearly trend
    })

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)

    # for i in range(16):
    #     X['txs_{}'.format(i)] = txs[t2017 + timedelta(days=i)].values

    # for i in range(1, 8):
    #     X["day_ron_{}".format(i)] = get_timespan(df_2017, t2017-timedelta(days=i), 1, 1).values.ravel()

    X = pd.concat([X, items_class], axis=1)

    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y

    return X


print("Preparing dataset...")
t2017 = date(2017, 5, 31)
#t2017 = date(2017, 4, 5)
X_l, y_l = [], []

periods = 6
for i in range(periods):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)


X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
X_val = X_val.as_matrix()
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

model = Sequential()
#model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=.1))
#model.add(Dropout(.1))
model.add(Dense(32))
model.add(Dropout(.1))
model.add(Dense(16))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

N_EPOCHS = 5

val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
sample_weights = np.array(pd.concat([items["perishable"]] * periods) * 0.25 + 1)
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    xv = X_val
    yv = y_val[:, i]
    model.fit(X_train, y, batch_size=512, epochs=N_EPOCHS, verbose=2,
              sample_weight=sample_weights, validation_data=(xv, yv))
    val_pred.append(model.predict(X_val))
    test_pred.append(model.predict(X_test))

n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).squeeze(axis=2).transpose()) )
print("Full validation mse:       ", mean_squared_error(
    y_val, np.array(val_pred).squeeze(axis=2).transpose(), sample_weight=weights) )
print("'Public' validation mse:   ", mean_squared_error(
    y_val[:,:n_public], np.array(val_pred).squeeze(axis=2).transpose()[:,:n_public], 
    sample_weight=weights) )
print("'Private' validation mse:  ", mean_squared_error(
    y_val[:,n_public:], np.array(val_pred).squeeze(axis=2).transpose()[:,n_public:], 
    sample_weight=weights) )
    
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=stores_items.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = test_ids.join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('../output/lstm.csv.gz', float_format='%.4f', index=None, compression="gzip")

