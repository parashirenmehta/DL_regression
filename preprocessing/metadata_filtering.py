import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/paras/eelgrass/metadata/EELgrass_Metadata.csv')
df = df.dropna(subset=['cover'])


df_imp = df[['image_filename', 'cover']]
df_imp.to_csv('../data/cover.csv', index=False)

print(df_imp.head())