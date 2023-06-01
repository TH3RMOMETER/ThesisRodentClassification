import os
from typing import List

from load_agouti import (
    convert_filtered_data_into_list_with_start_and_end_times,
    create_list_with_random_start_and_end_times,
    create_list_with_start_and_end_times,
    load_and_filter_agouti_data,
)
from config import Config
from process_audio import process_audio_MFCC, save_audio_file_start_end


def create_audio_points(config: Config, audio_file_path: str):
    """Create audio points from agouti data

    Args:
        config (Config): Config object with all variables

    Returns:
        (List, List): List with audio points from agouti data, list with random audio points
    """
    agouti = load_and_filter_agouti_data(
        media_filepath=config.agouti_media_path,
        observations_filepath=config.agouti_observations_path,
        audio_file_path=audio_file_path,
    )
    audio_points = convert_filtered_data_into_list_with_start_and_end_times(
        agouti_filtered=agouti, audio_file_path=audio_file_path
    )
    audio_points = create_list_with_start_and_end_times(
        audio_points=audio_points, difference=config.audio_length
    )
    random_audio_points = create_list_with_random_start_and_end_times(
        audio_ranges=audio_points,
        difference=config.audio_length,
        audio_path=audio_file_path,
    )
    return audio_points, random_audio_points


def create_mfcc_data(config: Config, audio_points: List, random_audio_points: List):
    """Create MFCC data from audio points

    Args:
        config (Config): Config object with all variables
        audio_points (List): list with audio points from agouti data
        random_audio_points (List): list with random audio points
    """
    for index, point in enumerate(audio_points):
        process_audio_MFCC(
            audio_path=config.audio_data_folder_path,
            start_time=point[0],
            end_time=point[1],
            slice_len=config.slice_len,
            step_size=config.step_size,
            output_path=f"{config.output_path}\\rats\\",
            index=index,
        )
    for index, point in enumerate(random_audio_points):
        process_audio_MFCC(
            audio_path=config.audio_data_folder_path,
            start_time=point[0],
            end_time=point[1],
            slice_len=config.slice_len,
            step_size=config.step_size,
            output_path=f"{config.output_path}\\noise\\",
            index=index,
        )


def create_list_with_audio_filepaths_in_folder(folder_path: str):
    """Create list with audio filepaths in folder

    Args:
        folder_path (str): path to folder with audio files

    Returns:
        List: list with audio filepaths
    """
    audio_filepaths = []
    # only include .flac files
    for file in os.listdir(folder_path):
        if file.endswith(".flac"):
            audio_filepaths.append(os.path.join(folder_path, file))
    return audio_filepaths


def save_audio_points_to_file(
    config: Config, audio_points: List, random_audio_points: List, file_path: str
):
    """Create MFCC data from audio points

    Args:
        config (Config): Config object with all variables
        audio_points (List): list with audio points from agouti data
        random_audio_points (List): list with random audio points
    """
    for index, point in enumerate(audio_points):
        save_audio_file_start_end(
            file_path=file_path,
            output_path=f"{config.cropped_audio_path}\\rats\\",
            index=index,
            start_time=point[0],
            end_time=point[1],
        )
    for index, point in enumerate(random_audio_points):
        save_audio_file_start_end(
            file_path=file_path,
            output_path=f"{config.cropped_audio_path}\\noise\\",
            index=index,
            start_time=point[0],
            end_time=point[1],
        )


def process_audio_folder(config: Config):
    audio_filepaths = create_list_with_audio_filepaths_in_folder(
        folder_path=config.audio_data_folder_path
    )
    #  for each filepath create mfcc data
    for filepath in audio_filepaths:
        audio_points, random_audio_points = create_audio_points(
            config=config, audio_file_path=filepath
        )
        save_audio_points_to_file(
            config=config,
            audio_points=audio_points,
            random_audio_points=random_audio_points,
            file_path=filepath,
        )


if __name__ == "__main__":
    process_audio_folder(config=Config()) # type: ignore
