import argparse
import os
from pytube import YouTube
import tqdm


if __name__ == '__main__':
    output_dir = '/home/nas3_userK/hojoonlee/projects/atari_100k/data/youtube'

    videos = [
        # montezuma
        {'game':'montezuma', 'idx':1, 'video_link':'sYbBgkP9aMo', 'clip_window': [0,  22, 480, 341], 'start_time': 0.0},
        {'game':'montezuma', 'idx':2, 'video_link':'6zXXZvVvTFs', 'clip_window': [35, 50, 445, 300], 'start_time': 0.6},
        {'game':'montezuma', 'idx':3, 'video_link':'SuZVyOlgVek', 'clip_window': [79, 18, 560, 360], 'start_time': 0.2},
        {'game':'montezuma', 'idx':4, 'video_link':'2AYaxTiWKoY', 'clip_window': [0,  13, 640, 335], 'start_time': 8.8},
        {'game':'montezuma', 'idx':5, 'video_link':'pF6xCZA72o0', 'clip_window': [20,  3, 620, 360], 'start_time': 24.1},
        # private-eye
        {'game':'private_eye', 'idx':1, 'video_link':'zfdov0gmPRM', 'clip_window': [382, 37, 1221, 681], 'start_time': 14.5},
        {'game':'private_eye', 'idx':2, 'video_link':'YvaSsTIbvfc', 'clip_window': [235, 64, 1043, 654], 'start_time': 7.6},
        {'game':'private_eye', 'idx':3, 'video_link':'3Rqxqrbi9KE', 'clip_window': [-59, -18, 1128, 765], 'start_time': 2.8},
        {'game':'private_eye', 'idx':4, 'video_link':'jt0YhI_CYUs', 'clip_window': [296, 0, 1280, 718], 'start_time': 4.3},
        # pitfall
        {'game':'pitfall', 'idx':1, 'video_link':'CkDllyETiBA', 'clip_window': [123, 26, 527, 341], 'start_time': 25.6},
        {'game':'pitfall', 'idx':2, 'video_link':'aAJzamWAFOE', 'clip_window': [-30, -11, 570, 381], 'start_time': 8.3},
        {'game':'pitfall', 'idx':3, 'video_link':'sH4UpYOeDFA', 'clip_window': [0, 44, 480, 359], 'start_time': 8.3},
        {'game':'pitfall', 'idx':4, 'video_link':'BPYsqj5N_0I', 'clip_window': [70, 21, 506, 359], 'start_time': 1.0},
    ]

    for video in tqdm.tqdm(videos):
        game = video['game']
        idx = str(video['idx'])
        video_link = 'https://youtu.be/' + video['video_link']
        output_path = output_dir + '/' + game

        yt = YouTube(video_link).streams.get_highest_resolution()
        yt.download(output_path = output_path, filename=idx + '.mp4')

