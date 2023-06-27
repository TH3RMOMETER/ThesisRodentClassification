import os
import random
import shutil
import tarfile
import time
import zipfile
from math import e

import numpy as np
import pandas as pd
import tqdm
from config import Config
from process_audio import (
    get_audio_file_length,
    get_site_id_from_audio_file,
    get_timestamp_from_audio_file,
    load_agouti_data,
    select_random_audio_point,
    save_audio_file_start_end,
)

path = r"G:\AudioFiles"
output_path = r"G:\AudioFiles\audio"
config = Config()


# Extracting tar.gz files
def extract_tar_files(path, output_path):
    tar_files = [file for file in os.listdir(path) if file.endswith(".tar.gz")]
    for file in tqdm.tqdm(tar_files):
        tar = tarfile.open(os.path.join(path, file), "r:gz")
        tar.extractall(output_path)
        tar.close()
        # after extracting, remove the tar file
        os.remove(os.path.join(path, file))


def move_flac_files(src, dest):
    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(src, topdown=False)):
        for filename in filenames:
            if filename.endswith(".flac"):
                src_file_path = os.path.join(dirpath, filename)
                dest_file_path = os.path.join(dest, filename)
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved file: {src_file_path} to {dest_file_path}")

        # After moving files, check if the directory is empty and delete if it is
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")


def process_zip_file(config, zip_path, output_path, agouti: pd.DataFrame):
    with zipfile.ZipFile(zip_path, "r") as myzip:
        for filename in tqdm.tqdm(myzip.namelist()):
            myzip.extract(filename, output_path)
            tar_path = os.path.join(output_path, filename)
            # Open the .tar.gz file and extract its contents
            with tarfile.open(tar_path) as mytar:
                mytar.extractall(path=config.audio_folder)
            # move files from subfolder to output_path
            move_flac_files(config.audio_folder, config.audio_folder)
            # Delete the .tar.gz file after extraction
            os.remove(tar_path)
            print(f"Removed file: {tar_path}")
            # loop over all the flac files
            for dirpath, dirnames, filenames in tqdm.tqdm(
                os.walk(config.audio_folder, topdown=False)
            ):
                # loop over all the files in the directory
                for filename in filenames:
                    # if the file is a flac file
                    if filename.endswith(".flac"):
                        # process the file
                        audio_length = get_audio_file_length(
                            os.path.join(dirpath, filename)
                        )
                        start_time = get_timestamp_from_audio_file(filename)
                        end_time = start_time + pd.Timedelta(seconds=audio_length)
                        site_id = get_site_id_from_audio_file(filename)
                        site_id = site_id + "_"
                        # filter agouti on site_id and start_time and end_time
                        agouti_filtered = agouti.query(
                            "fileName.str.contains(@site_id) & timestamp >= @start_time and timestamp <= @end_time"
                        )
                        if not agouti_filtered.empty:
                            for index, row in tqdm.tqdm(
                                agouti_filtered.iterrows(),
                                total=agouti_filtered.shape[0],
                            ):
                                # get timestamp of instance
                                timestamp = row["timestamp"]
                                # get random timestamp
                                random_timestamp = select_random_audio_point(timestamp)
                                # check if random timestamp is within audio file
                                loop_counter = 0
                                loop_check = False
                                while (
                                    random_timestamp
                                    - pd.Timedelta(seconds=config.audio_length / 2)
                                    < start_time
                                    or random_timestamp
                                    + pd.Timedelta(seconds=config.audio_length / 2)
                                    > end_time
                                ):
                                    if loop_counter > 200:
                                        loop_check = True
                                        continue
                                    random_timestamp = select_random_audio_point(
                                        timestamp
                                    )
                                    loop_counter += 1
                                if loop_check:
                                    continue
                                # convert start and end time to seconds from start of audio file
                                start_time_seconds = (
                                    timestamp - start_time
                                ).total_seconds()
                                end_time_seconds = (
                                    timestamp + pd.Timedelta(seconds=audio_length)
                                    - start_time
                                ).total_seconds()
                                random_start_time_seconds = (
                                    random_timestamp - start_time
                                ).total_seconds()
                                random_end_time_seconds = (
                                    random_timestamp + pd.Timedelta(seconds=audio_length)
                                    - start_time
                                ).total_seconds()
                                # check if start and end time are within audio file
                                if (
                                    start_time_seconds < 0
                                    or end_time_seconds > audio_length
                                ):
                                    continue
                                if (
                                    random_start_time_seconds < 0
                                    or random_end_time_seconds > audio_length
                                ):
                                    continue                                

                                # save cropped audio file
                                save_audio_file_start_end(
                                    output_path=os.path.join(
                                        config.cropped_audio_path, "rats"
                                    ),
                                    start_time=int(start_time_seconds),
                                    end_time=int(end_time_seconds),
                                    file_path=os.path.join(dirpath, filename),
                                    index=loop_counter,
                                )
                                save_audio_file_start_end(
                                    output_path=os.path.join(
                                        config.cropped_audio_path, "noise"
                                    ),
                                    start_time=int(random_start_time_seconds),
                                    end_time=int(random_end_time_seconds),
                                    file_path=os.path.join(dirpath, filename),
                                    index=loop_counter,
                                )
                                print(f'saved cropped audio file: {os.path.join(config.cropped_audio_path, "rats")} fpr site_id: {site_id} and timestamp: {timestamp}')
                        # when processed remove the file
                        os.remove(os.path.join(dirpath, filename))
                        print(f"Removed file: {os.path.join(dirpath, filename)}")
                    # if the file is not a flac file, remove
                    else:
                        # remove the file
                        os.remove(os.path.join(dirpath, filename))
                        print(f"Removed file: {os.path.join(dirpath, filename)}")
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
                    print(f"Removed empty directory: {dirpath}")


def return_audio_file_and_remove_rest(audio_file_names: list, audio_folder):
    """Remove all files from the audio_folder except the ones in audio_file_names"""
    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(audio_folder, topdown=False)):
        for filename in filenames:
            # remove extension from filename
            filename_no_extension = os.path.splitext(filename)[0]
            if filename_no_extension not in audio_file_names:
                os.remove(os.path.join(dirpath, filename))
                print(f"Removed file: {os.path.join(dirpath, filename)}")
        # After moving files, check if the directory is empty and delete if it is
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")


if __name__ == "__main__":
    config = Config()
    agouti = load_agouti_data(config)
    # check run time
    start_time = time.time()
    process_zip_file(config, config.zip_path, config.audio_folder, agouti)
    print(f"Run time: {time.time() - start_time}")
