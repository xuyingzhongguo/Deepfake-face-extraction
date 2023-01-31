import pandas as pd
from facenet_pytorch import MTCNN, extract_face
from PIL import Image
import mmcv
import cv2
from tqdm import tqdm
import os

image_size = 512
margin = 120
data_path = '../DFDC/train'
output_path = '../DFDC/faces'
mtcnn = MTCNN(image_size=image_size, margin=margin, select_largest=False, keep_all=True, device='cuda:0')


def main():
    # df_videos = pd.read_pickle('/home/ying/code/icpr2020dfdc/data/dfdc_videos_1.pkl')
    df_videos = pd.read_pickle('test_video.pkl')

    for idx, record in tqdm(df_videos.iterrows(), total=len(df_videos), desc='Collecting faces results'):
        video_path = os.path.join(data_path, record['path'])
        label = 1 if record['label'] else 0
        image_name = record['path'][-14:-4]
        try:
            process_video(video_path, label, image_name)
        except Exception as e:
            print('Error while reading: {}'.format(video_path))
            print(e)
        # process_video(video_path, label, image_name)


def process_video(video_path, label, image_name):
    video = mmcv.VideoReader(video_path)
    save_path = os.path.join(output_path, str(label), image_name)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    for i, frame in enumerate(frames):
        # print('\rTracking frame: {}'.format(i + 1), end='')
        if i % 10 == 1:
            boxes, probs = mtcnn.detect(frame, landmarks=False)
            if boxes is not None:
                for j, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > 0.98:
                        extract_face(frame, box, image_size=image_size, margin=margin, save_path=f'{save_path}/frame{i}_{j}_{prob}.png')


if __name__ == '__main__':
    main()
