import pandas as pd
from facenet_pytorch import MTCNN, extract_face
from PIL import Image
import mmcv
import cv2
from tqdm import tqdm
import os
import pickle5

image_size = 512
margin = 120
output_path = 'Celeb-DF-v2_faces_single'
mtcnn = MTCNN(image_size=image_size, margin=margin, select_largest=False, keep_all=False, device='cuda:0')


def main():
    df_videos = pd.read_pickle('Cele-DF-v2.pkl')
    # df_videos = pd.read_pickle('test_celeb.pkl')

    for idx, record in tqdm(df_videos.iterrows(), total=len(df_videos), desc='Collecting faces results'):
        video_path = os.path.join(record['folder'], record['folder']
        middle = record['path'].split('/')
        folder_name = os.path.join(middle[1], middle[2][:-4])
        try:
            process_video(video_path, folder_name)
        except Exception as e:
            print('Error while reading: {}'.format(video_path))
            print(e)
        # process_video(video_path, label, image_name)


def process_video(video_path, folder_name):
    video = mmcv.VideoReader(video_path)
    save_path = os.path.join(output_path, folder_name)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    for i, frame in enumerate(frames):
        # print('\rTracking frame: {}'.format(i + 1), end='')
        if i % 10 == 1 and i < 301:
            box, prob = mtcnn.detect(frame, landmarks=False)
            if box is not None and float(prob[0]) > 0.99:
                extract_face(frame, box[0], image_size=image_size, margin=margin, save_path=f'{save_path}/frame{i}_{str(prob[0])[:8]}.png')


if __name__ == '__main__':
    main()