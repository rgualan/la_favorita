#import itertools

import pandas as pd
import numpy as np
import time, sys

# print("Reading input file...")
# st = time.time()
# df_train = pd.read_csv(
#     'input/train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},
#     converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
#     skiprows=range(1, 124035460)
# )
# print("--- {} seconds ---".format(time.time() - st))
# df_train.to_csv("input/train2017.csv")

print("Reading (incomplete) input file...")
st = time.time()
df_train = pd.read_csv(
    'input/train2017.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0} #TODO: NaN could be 0 or unknown
)
print("--- {0:.2f} seconds ---".format(time.time() - st))
#print(df_train.head())

# log transform
df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p)

# Fill gaps in dates
# Improved with the suggestion from Paulo Pinto
u_dates = df_train.date.unique()
u_stores = df_train.store_nbr.unique()
u_items = df_train.item_nbr.unique()
df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
df_train = df_train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date", "store_nbr", "item_nbr"]
    )
)
#print(df_train.head())

# Fill NAs
# Assume missing unit_sales imply no sales
df_train.loc[:, "unit_sales"].fillna(0, inplace=True) ## TODO: Is this the best method?
# Assume missing entries imply no promotion
df_train.loc[:, "onpromotion"].fillna("False", inplace=True)

# Calculate means 
df_train = df_train.groupby(
    ['item_nbr', 'store_nbr', 'onpromotion']
)['unit_sales'].mean().to_frame('unit_sales')
#print(df_train.head())

# Inverse transform
df_train["unit_sales"] = df_train["unit_sales"].apply(np.expm1)
#print(df_train.head())

# Create submission
pd.read_csv(
    "input/test.csv", usecols=[0, 2, 3, 4], dtype={'onpromotion': str}
).set_index(
    ['item_nbr', 'store_nbr', 'onpromotion']
).join(
    df_train, how='left'
).fillna(0).to_csv(
    'output/mean.csv.gz', float_format='%.2f', index=None, compression="gzip"
)

