"""Script for annotating videos"""
import os
import glob
import subprocess
from pathlib import Path


def compress_video(video_in, video_out):
    subprocess.run(f'ffmpeg -n -i {video_in} -filter:v scale=1920x1080 -filter:v fps=fps=1 {video_out}', shell=True)


def perform_sem_seg_on_video(video_in, video_out):
    subprocess.run(f'python3 demo.py '
                   f'--config-file ../configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml '
                   f'--video-input {video_in} '
                   f'--output '
                   f'{video_out} '
                   f'--opts MODEL.WEIGHTS ../../../../Models/hrnet-mappilary/model_final_f3fc73.pkl',
                   shell=True)


def alter_name_for_sem_seg_video(video_base_name):
    pref, suf = video_base_name.split('.')
    pref += '_segmented'
    return '.'.join([pref, suf])


def main():
    input_dir = '/home/integrant/Documents/ucLh/Programming/Python/Datasets/ready_videos'
    out_dir = '/home/integrant/Documents/ucLh/Programming/Python/Datasets/ready_videos_compressed'
    video_names = glob.glob(f'{input_dir}/*/*.mp4')
    video_names = sorted(video_names)
    for video in video_names:
        base_folder = video.split('/')[-2]
        base_name = video.split('/')[-1]
        compressed_video = os.path.join(out_dir, base_folder, base_name)
        out_video_dir = os.path.dirname(compressed_video)
        Path(out_video_dir).mkdir(exist_ok=True, parents=True)
        compress_video(video, compressed_video)

        sem_seg_video = os.path.join(out_dir, base_folder, alter_name_for_sem_seg_video(base_name))
        perform_sem_seg_on_video(compressed_video, sem_seg_video)


if __name__ == '__main__':
    main()

