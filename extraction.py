import os
import re
import pandas as pd

downloads_df = None

for test in os.listdir('./data/raw'):
  if not os.path.exists(f'./data/raw/{test}/out.txt'):
    print(f'No out.txt in {test}')
    continue
  with open(f'./data/raw/{test}/out.txt', 'r') as f:
    lines = f.readlines()
    download_line = [line for line in lines if line.strip().startswith('Throughput:') ][0].strip()
    match = re.search(r'Throughput:\s+(\d+(\.\d+)?)', download_line)
    if match:
      throughput = float(match.group(1))
      df = pd.DataFrame({'test': test, 'download_mbps': throughput}, index=[0])
      if downloads_df is None:
        downloads_df = df
      else:
        downloads_df = pd.concat([downloads_df, df], ignore_index=True)
    else:
      print(f'No throughput found in {test}')
      
downloads_df.to_csv('./data/speeds.csv', index=False)