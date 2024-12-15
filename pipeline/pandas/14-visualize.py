#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df.set_index('Date', inplace=True)

df['Close'] = df['Close'].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df = df[df.index >= '2017']

df_resampled = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

df_resampled[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']].plot(figsize=(10, 6))
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('plot.png')
plt.close()
