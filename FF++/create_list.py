import numpy as np
from pathlib import Path
import pandas as pd

folder_paths = [
                '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/Deepfakes/c40/videos',
                '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/Face2Face/c40/videos',
                '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/FaceSwap/c40/videos',
                '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/NeuralTextures/c40/videos',
                '/cluster/home/xuyi/xuyi/FF++/original_sequences/youtube/c40/videos',
                ]

# '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/Deepfakes/c40/videos',

video_lists = pd.Series([])

for item in folder_paths:
    path = Path(item)
    video_list = pd.Series(path.glob('*.mp4'))
    video_lists = video_lists.append(video_list, ignore_index=True)

df = pd.DataFrame(video_lists, columns=['path'])
df['path'] = df['path'].astype(str)
df['label'] = [1 if 'manipulated' in x else 0 for x in df['path']]

df.to_pickle('FF++_c40_list.pkl')

# object = pd.read_pickle('FF++_c40_list.pkl')
#
# object = object.loc[:10, :]
# print(object.iloc[0, 0].strip('''''').split('/'))
# object.to_pickle('test_c40.pkl')
