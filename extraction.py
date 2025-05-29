import os
import pandas as pd
import json
import gzip

def generate_from_server_measurements():
  speeds_df = None
  valid_tests = 0

  for test in os.listdir('./data'):
    if not os.path.exists(f'./data/{test}/ndt7'):
      print(f'Skipping {test}, no ndt7 directory found')
      continue
    for ndt7_file in os.listdir(f'./data/{test}/ndt7'):
      if ndt7_file.startswith('ndt7-download') and ndt7_file.endswith('.json') or ndt7_file.endswith('.json.gz'):
        compressed = ndt7_file.endswith('.json.gz')
        filename = ndt7_file
        with (gzip.open(f'./data/{test}/ndt7/{filename}', 'rt') if compressed else open(f'./data/{test}/ndt7/{filename}', 'r')) as f:
          data = json.load(f)
          if not data.get('Download') or not data.get('Download').get('ServerMeasurements'):
            continue
          measurements = data['Download']['ServerMeasurements']
          last_measurement = measurements[-1]
          speed = last_measurement['TCPInfo']['BytesAcked'] / last_measurement['TCPInfo']['ElapsedTime'] * 8
          if speed is None:
            print(f'No speed data found in {test}')
            continue
          
          valid_tests += 1
          for measurement in measurements:
            row = {
              'TestID': valid_tests,
              'ElapsedTime': measurement['TCPInfo']['ElapsedTime'],
              'BytesSent': measurement['TCPInfo']['BytesSent'],
              'BytesAcked': measurement['TCPInfo']['BytesAcked'],
              'BytesRetrans': measurement['TCPInfo']['BytesRetrans'],
              'RTT': measurement['TCPInfo']['RTT'],
              'RTTVar': measurement['TCPInfo']['RTTVar'],
              'RWndLimited': measurement['TCPInfo']['RWndLimited'],
              'SndBufLimited': measurement['TCPInfo']['SndBufLimited'],
              'MinRTT': measurement['TCPInfo']['MinRTT'],
              'FinalSpeed': speed
            }
            df = pd.DataFrame([row])
            if speeds_df is None:
              speeds_df = df
            else:
              speeds_df = pd.concat([speeds_df, df], ignore_index=True)
          
  print(f'{valid_tests} unique tests found with valid speed data')
  speeds_df.to_csv('./dataset.csv', index=False)
  
def generate_from_tcpinfo_traces():
  speeds_df = None
  valid_tests = 0

  for test in os.listdir('./data'):
    if not os.path.exists(f'./data/{test}/tcpinfo'):
      print(f'Skipping {test}, no tcpinfo directory found')
      continue
    if not os.path.exists(f'./data/{test}/tcpinfo/trace.csv'):
      print(f'Skipping {test}, no trace.csv file found')
      continue
    valid_tests += 1
    trace_df = pd.read_csv(f'./data/{test}/tcpinfo/trace.csv').iloc[1:]
    trace_df['TestID'] = valid_tests
    trace_df['ElapsedTime'] = ((pd.to_datetime(trace_df['Timestamp']) - pd.to_datetime(trace_df['Timestamp'].iloc[0])).dt.total_seconds() * 1_000_000).astype(int)
    trace_df['FinalSpeed'] = trace_df['TCP.BytesAcked'].astype(float).iloc[-1] / trace_df['ElapsedTime'].astype(float).iloc[-1] * 8
    df = trace_df[['TestID', 'ElapsedTime']].copy()
    df['BytesSent'] = trace_df['TCP.BytesSent'].astype(int)
    df['BytesAcked'] = trace_df['TCP.BytesAcked'].astype(int)
    df['BytesRetrans'] = trace_df['TCP.BytesRetrans'].astype(int)
    df['RTT'] = trace_df['TCP.RTT'].astype(int)
    df['RTTVar'] = trace_df['TCP.RTTVar'].astype(int)
    df['RWndLimited'] = trace_df['TCP.RWndLimited'].astype(int)
    df['SndBufLimited'] = trace_df['TCP.SndBufLimited'].astype(int)
    df['MinRTT'] = trace_df['TCP.MinRTT'].astype(int)
    df['FinalSpeed'] = trace_df['FinalSpeed'].astype(float)
    df = df.iloc[1:]
    if speeds_df is None:
      speeds_df = df
    else:
      speeds_df = pd.concat([speeds_df, df], ignore_index=True)
          
  print(f'{valid_tests} unique tests found with valid speed data')
  speeds_df.to_csv('./dataset.csv', index=False)

if __name__ == '__main__':
  generate_from_tcpinfo_traces()