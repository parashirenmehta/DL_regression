import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/cover.csv')
df = df[df['cover'] != 0]
df.hist(column='cover', bins=5)
plt.xlabel('Cover')
plt.ylabel('Frequency')
plt.show()

