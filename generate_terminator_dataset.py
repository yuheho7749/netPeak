import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from predictor import ThroughputPredictor

WINDOW_SIZE_MS = 100 # Number of milliseconds for each time window; TCP_INFO statistics are aggregated over this period
TIME_SERIES_LENGTH = 5 # Number of time series steps to consider for prediction

def load_dataset(file_path):
  df = pd.read_csv(file_path).dropna()
  df.sort_values(by=['TestID', 'ElapsedTime'], inplace=True)
  
  df['k'] = df['ElapsedTime'] // (WINDOW_SIZE_MS * 1000)
  agg_df = df.groupby(['TestID', 'k'], as_index=False).agg({
    'ElapsedTime': 'max',
    'BusyTime': 'max',
    'BytesSent': 'max',
    'BytesAcked': 'max',
    'BytesRetrans': 'max',
    'RTT': 'mean',
    'RTTVar': 'mean',
    'MinRTT': 'min',
    'RWndLimited': 'max',
    'SndBufLimited': 'max',
    'FinalSpeed': 'max',
  }, inplace=True)
  
  for i in range(1, TIME_SERIES_LENGTH):
    agg_df[f'ElapsedTime_{i}'] = agg_df.groupby('TestID')['ElapsedTime'].shift(i).fillna(0)
    agg_df[f'BusyTime_{i}'] = agg_df.groupby('TestID')['BusyTime'].shift(i).fillna(0)
    agg_df[f'BytesSent_{i}'] = agg_df.groupby('TestID')['BytesSent'].shift(i).fillna(0)
    agg_df[f'BytesAcked_{i}'] = agg_df.groupby('TestID')['BytesAcked'].shift(i).fillna(0)
    agg_df[f'BytesRetrans_{i}'] = agg_df.groupby('TestID')['BytesRetrans'].shift(i).fillna(0)
    agg_df[f'RTT_{i}'] = agg_df.groupby('TestID')['RTT'].shift(i).fillna(0)
    agg_df[f'RTTVar_{i}'] = agg_df.groupby('TestID')['RTTVar'].shift(i).fillna(0)
    agg_df[f'RWndLimited_{i}'] = agg_df.groupby('TestID')['RWndLimited'].shift(i).fillna(0)
    agg_df[f'SndBufLimited_{i}'] = agg_df.groupby('TestID')['SndBufLimited'].shift(i).fillna(0)
    
  agg_df = agg_df.groupby('TestID').apply(lambda x: x.iloc[TIME_SERIES_LENGTH:]).reset_index(drop=True)
  agg_df.drop(columns=['TestID'], inplace=True)
  labels = agg_df.pop('FinalSpeed')
  
  return agg_df, labels

features, labels = load_dataset('./dataset.csv')
print(features)

X, y = features.values, labels.values
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression = ThroughputPredictor(num_features=len(features.columns))
regression.to(device)
regression.fit(X_train, y_train, epochs=1000, batch_size=16, learning_rate=0.001)
# regression.fit(X_train, y_train, epochs=50, batch_size=16, learning_rate=0.01)

X = scaler.fit_transform(X)
y_pred = regression.predict(X)
percent_error = np.abs((y_pred - y) / y)
# print(percent_error)
# print(len(percent_error))

# df = features[["k"]].copy()
df = features.copy()
df["EstimatedSpeed"] = y_pred
# df["FinalSpeed"] = labels
# TODO: (1 - percent_error) to match sigmoid function
# 1 mean the predictor should stop
# df["PercentError"] = percent_error
# print(df)

df.to_csv("./terminator_dataset.csv", index=False)

# ax = plt.axes()
# ax.ecdf(percent_error, label='Percent Error')
# ax.set_xlabel('Percent Error (%)')
# ax.set_ylabel('Proportion of Samples')
# plt.xlim(0, 100)
# plt.show()
#
