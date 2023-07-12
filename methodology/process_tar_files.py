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


def process_zip_file(
    config: Config, zip_path: str, output_path: str, agouti: pd.DataFrame
) -> None:
    """Process the zip file and extract the audio files

    Args:
        config (Config): configuration object
        zip_path (str): path to the zip file
        output_path (str): path to the output folder
        agouti (pd.DataFrame): agouti dataframe
    """
    setups = pd.read_pickle(r"G:\thesis\ThesisRodentClassification\setups.pkl")
    arise = pd.read_pickle(
        r"G:\thesis\ThesisRodentClassification\pandas_df_arise_api_gijs_data_request1.pkl"
    )
    arise["recording_dt"] = pd.to_datetime(arise["recording_dt"], utc=True)
    in_setups = []
    in_arise = []
    total = []
    no_match_df = pd.DataFrame()
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
                        # for testing purposes, if no match is found, check the nearest match and add to dataframe
                        start_time = get_timestamp_from_audio_file(filename)
                        end_time = start_time + pd.Timedelta(seconds=audio_length)
                        site_id = get_site_id_from_audio_file(filename)
                        site_id = site_id + "_"
                        # filter agouti on site_id and start_time and end_time
                        agouti_filtered = agouti.query(
                            "fileName.str.contains(@site_id) & timestamp >= @start_time and timestamp <= @end_time"
                        )
                        # check if site id and date is in setups
                        setups_filtered = setups[
                            setups.str.contains(site_id)
                            & setups.str.contains(start_time.strftime("%Y-%m-%d"))
                        ]
                        if not setups_filtered.empty:
                            in_setups.append(filename)
                        # check if site id and date is in arise
                        arise_filtered = arise[
                            arise.deployment.str.contains(site_id)
                            & arise.recording_dt.dt.date.eq(start_time.date()
                            )
                        ]
                        # check if site_id and date is in arise
                        if not arise_filtered.empty:
                            in_arise.append(filename)

                        total.append(filename)
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

                                # get time difference in seconds between start and timestamp
                                time_difference = (
                                    timestamp - start_time
                                ).total_seconds()
                                # get time difference in seconds between start and random timestamp
                                random_time_difference = (
                                    random_timestamp - start_time
                                ).total_seconds()
                                # create start and end time for cropped audio file
                                start_time_seconds = (
                                    time_difference - config.audio_length / 2
                                )
                                end_time_seconds = (
                                    time_difference + config.audio_length / 2
                                )
                                random_start_time_seconds = (
                                    random_time_difference - config.audio_length / 2
                                )
                                random_end_time_seconds = (
                                    random_time_difference + config.audio_length / 2
                                )
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
                                print(
                                    f'saved cropped audio file: {os.path.join(config.cropped_audio_path, "rats")} for site_id: {site_id} and timestamp: {timestamp}'
                                )

                        else:
                            # if no match is found, check the nearest match and add to dataframe
                            start_time = get_timestamp_from_audio_file(filename)
                            end_time = start_time + pd.Timedelta(seconds=audio_length)
                            # get average time from start and end
                            average_time = start_time + pd.Timedelta(seconds=audio_length / 2)
                            site_id = get_site_id_from_audio_file(filename)

                            # check agouti df for site_id
                            agouti_filtered = agouti.query(
                                "fileName.str.contains(@site_id)"
                            )
                            if not agouti_filtered.empty:
                                # check nearest timestamp
                                nearest_timestamp = min(
                                    agouti_filtered["timestamp"],
                                    key=lambda x: abs(x - average_time),
                                )
                                # get time difference in seconds between average time and nearest timestamp
                                time_difference = (
                                    nearest_timestamp - average_time
                                ).total_seconds()
                                # add filename and time difference to dataframe
                                no_match_df = no_match_df.append(
                                    {
                                        "filename": filename,
                                        "time_difference": time_difference,
                                    },
                                    ignore_index=True,
                                ) # type: ignore


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
    # export in_setups and in_arise and total to csv
    pd.DataFrame(in_setups).to_csv(
        r"G:\thesis\ThesisRodentClassification\in_setups.csv"
    )
    pd.DataFrame(in_arise).to_csv(r"G:\thesis\ThesisRodentClassification\in_arise.csv")
    pd.DataFrame(total).to_csv(r"G:\thesis\ThesisRodentClassification\total.csv")
    no_match_df.to_csv(r"G:\thesis\ThesisRodentClassification\no_match_df.csv")


def check_arise_df(arise_df: pd.DataFrame, config: Config, agouti_df: pd.DataFrame) -> pd.DataFrame:
    """Check how many matches are found in the arise dataframe

    Args:
        arise_df (pd.DataFrame): _description_
        config (Config): _description_
        agouti_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    match_df = pd.DataFrame()
    total_datapoints = 0
    # iterate over arise dataframe
    for index, row in tqdm.tqdm(arise_df.iterrows(), total=arise_df.shape[0]):
        filename = row["filename"]
        # get start time of audio file
        start_time = get_timestamp_from_audio_file(filename)
        # get duration of audio file
        audio_length = int(row.extrainfo['duration'])
        # get end time of audio file
        end_time = start_time + pd.Timedelta(seconds=audio_length)
        # get site id of audio file
        site_id = get_site_id_from_audio_file(filename)
        site_id = site_id + "_"
        # filter agouti data
        agouti_filtered = agouti_df.query(
            "fileName.str.contains(@site_id) & timestamp >= @start_time and timestamp <= @end_time"
        )
        # add matches to match dataframe
        if not agouti_filtered.empty:
            match_df = match_df.append(
                {
                    "filename": filename,
                    "start_time": start_time,
                    "end_time": end_time,
                    "site_id": site_id,
                    "agouti_matches": len(agouti_filtered),
                },
                ignore_index=True,
            ) # type: ignore
        total_datapoints += len(agouti_filtered)
    return match_df        

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
    """
    # check run time
    start_time = time.time()
    process_zip_file(config, config.zip_path, config.audio_folder, agouti)
    print(f"Run time: {time.time() - start_time}") """
    arise_df = pd.read_pickle(r'G:\thesis\ThesisRodentClassification\gijs_datarequest_arise_full.pkl')
    match_df = check_arise_df(arise_df=arise_df, config=config, agouti_df=agouti)
    x=1
