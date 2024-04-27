import os
import threading
import multiprocessing
from queue import Queue
from pathlib import Path
from tqdm import tqdm
import imageio

"""
Given individual video files (mp4, webm) on disk, creates a folder for
every video file and saves the video's RGB frames as jpeg files in that
folder.

Uses multithreading to extract frames faster.

Modify the two filepaths at the bottom and then run this script.

Source: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
"""


def video_to_rgb(video_filepath: Path, out_dir: Path):
    reader = imageio.get_reader(video_filepath)

    if len(reader) > len(os.listdir(out_dir)):
        for frame_number, im in enumerate(reader):
            out_filepath = out_dir / f'frame_{frame_number}.jpg'
            if not out_filepath.is_file():
                imageio.imwrite(out_filepath, im)


def process_videofile(
        video_filename: str,
        video_dir_path: Path,
        rgb_out_path: Path,
        file_extension='.mp4',
        delete_video=False,
):
    filepath = video_dir_path / video_filename
    video_filename = video_filename.replace(file_extension, '')

    out_dir = rgb_out_path / video_filename
    out_dir.mkdir(parents=True, exist_ok=True)
    video_to_rgb(filepath, out_dir)
    if delete_video:
        filepath.unlink()


def thread_job(queue, video_path, rgb_out_path, file_extension='.mp4'):
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(
            video_filename,
            video_path,
            rgb_out_path,
            file_extension=file_extension,
            delete_video=False,
        )
        queue.task_done()


if __name__ == '__main__':
    script_dir = Path(os.path.realpath(os.path.dirname(__file__)))
    # the path to the folder which contains all video files (mp4, webm, or other)
    video_path = script_dir / "videos"
    # the root output path where RGB frame folders should be created
    rgb_out_path = script_dir / "images"
    # the file extension that the videos have
    file_extension = '.mp4'
    # height and width to resize RGB frames to

    video_filenames = os.listdir(video_path)
    queue = Queue()
    [queue.put(video_filename) for video_filename in video_filenames]

    num_threads = multiprocessing.cpu_count()

    for i in tqdm(range(num_threads)):
        worker = threading.Thread(target=thread_job,
                                  args=(queue,
                                        video_path,
                                        rgb_out_path,
                                        file_extension
                                        )
                                  )
        worker.start()

    print('Waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take more then an hour depending on dataset size')
    queue.join()
    print('Done')
