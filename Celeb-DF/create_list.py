import numpy as np
from pathlib import Path
import pandas as pd

folder_paths = ['../Celeb-real',
                '../Celeb-synthesis',
                '../YouTube-real', ]

video_lists = pd.Series([])

for item in folder_paths:
    path = Path(item)
    video_list = pd.Series(path.glob('*.mp4'))
    video_lists = video_lists.append(video_list, ignore_index=True)

df = pd.DataFrame(video_lists, columns=['path'])
df['path'] = df['path'].astype(str)
df['label'] = [1 if 'synthesis' in x else 0 for x in df['path']]

df.to_pickle('Cele-DF-v2.pkl')
