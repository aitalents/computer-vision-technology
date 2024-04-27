import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

LINK_BASE = "https://youtu.be/"


def download_video_by_id(yt_id: str, save_dir: Path, save_filename: str):
    link = LINK_BASE + yt_id
    yt = YouTube(link)
    yt.streams.filter(progressive=True, file_extension="mp4").order_by(
        "resolution"
    ).desc().first().download(output_path=save_dir, filename=f"{save_filename}.mp4")


def crop_video(
    in_path: Path, out_path: Path, start_time: float, end_time: float, delete_orig=False
):
    sys.stdout = open(os.devnull, "w")
    ffmpeg_extract_subclip(
        filename=in_path, t1=start_time, t2=end_time, targetname=out_path
    )

    if delete_orig:
        in_path.unlink()
    sys.stdout = sys.__stdout__


def main(pattern="dancing"):
    script_dir = Path(os.path.realpath(os.path.dirname(__file__)))
    # out_dir = "/kaggle/working/"
    out_dir = Path("/kaggle/working/")
    # save_videos_dir = os.path.join(out_dir, "videos")
    save_videos_dir = out_dir / "videos"

    save_videos_dir.mkdir(parents=True, exist_ok=True)

    for stage in ["train", "validate"]:
        labels_filename = f"{stage}.csv"
        df = pd.read_csv(script_dir / labels_filename)
        df = df[df["label"].str.contains(pattern)]
        target_df = df

        for index, row in tqdm(
            target_df.iterrows(), f"Downloading {stage}", total=len(df.index)
        ):
            yt_id = row["youtube_id"]
            start_time = float(row["time_start"])
            end_time = float(row["time_end"])
            full_video_name = f"tmp_{yt_id}"
            # full_video_path = os.path.join(save_videos_dir, f"/{full_video_name}.mp4")
            # cropped_video_path = os.path.join(save_videos_dir, f"/{yt_id}.mp4")
            full_video_path = save_videos_dir / f"{full_video_name}.mp4"
            cropped_video_path = save_videos_dir / f"{yt_id}.mp4"

            if not cropped_video_path.is_file():
                try:
                    download_video_by_id(
                        yt_id=yt_id,
                        save_dir=save_videos_dir,
                        save_filename=full_video_name,
                    )
                    crop_video(
                        in_path=full_video_path,
                        out_path=cropped_video_path,
                        start_time=start_time,
                        end_time=end_time,
                        delete_orig=True,
                    )
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    print(f"Error processing {yt_id}: {e}")
                    target_df = target_df.drop([index])

        target_df = target_df.drop(["time_start", "time_end", "split"], axis=1)
        target_df = target_df.rename(columns={"Unnamed: 0": "orig_id"})
        save_csv_name = f"{pattern}-{stage}.csv"
        # target_df.to_csv(os.path.join(out_dir, save_csv_name))
        target_df.to_csv(out_dir / save_csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Kinetics 700-2020 videos downloader",
        description="Download videos from YouTube for Kinetics 700-2020 "
        "dataset with any pattern and save new .csv labels",
    )
    parser.add_argument(
        "pattern",
        type=str,
        help="label pattern (example: 'dancing' for download all dancing videos)",
    )
    args = parser.parse_args()

    main(pattern=args.pattern)
