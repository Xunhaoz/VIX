import utils

df = utils.get_stock(['TSM', "^TWII"])
df = df[['Adj Close']]
df_valid_index = df.first_valid_index()
print(df_valid_index)