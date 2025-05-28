import os
import pandas as pd
import json
import gzip

speeds_df = None
valid_tests = 0

for test in os.listdir('./data'):
  if not os.path.exists(f'./data/{test}/ndt7'):
    print(f'Skipping {test}, no ndt7 directory found')
    continue
  filename = None
  compressed = False
  for ndt7_file in os.listdir(f'./data/{test}/ndt7'):
    if ndt7_file.startswith('ndt7-download') and ndt7_file.endswith('.json') or ndt7_file.endswith('.json.gz'):
      compressed = ndt7_file.endswith('.json.gz')
      filename = ndt7_file
      break
  if filename is None:
    print(f'No ndt7-download file found in {test}')
    continue
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
        
print(f'Valid: {valid_tests}, Total: {len(os.listdir("./data"))}')
speeds_df.to_csv('./dataset.csv', index=False)