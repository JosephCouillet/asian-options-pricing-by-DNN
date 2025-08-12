import pandas as pd

df = pd.read_csv('data/pricing_dataset.csv')
assert not df.isnull().any().any()        # no NaN
assert df.shape[0] == 100_000               # good number of rows
assert df['S0'].between(50,150).all()
assert (df['price']>=0).all()
assert df['price'].max() <= df['S0'].max()

