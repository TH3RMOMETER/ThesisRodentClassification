import random
from datetime import datetime

import audioread
import numpy as np
import pandas as pd
import pytz

from methodology.process_audio import get_audio_file_length, get_sample_rate, convert_seconds_to_frames_in_audiofile


def load_agouti_data(media_filepath: str, observations_filepath: str) -> pd.DataFrame:
    """
    Load in the agouti data, merge the observations and media files and remove duplicates.
    """
    # load in the data
    agouti_media = pd.read_csv(media_filepath)
    agouti_observations = pd.read_csv(observations_filepath)

    # remove duplicates from agouti_media (multiple photos of same instance)
    agouti_media = agouti_media.drop_duplicates(subset=['sequenceID'])

    agouti = pd.merge(agouti_observations, agouti_media, on=[
        'sequenceID', 'deploymentID', 'timestamp'], suffixes=('_obs', '_media')) \
        .dropna(axis=1, how='all')

    # convert timestamp to datetime object
    # if flevopark is in the filepath, the timestamp is in UTC + 1
    if 'flevopark' in media_filepath:
        agouti.timestamp = pd.to_datetime(agouti.timestamp, utc=True).map(lambda x: x.tz_convert('Etc/GMT-1'))
    else:
        agouti.timestamp = pd.to_datetime(agouti.timestamp, utc=True).map(lambda x: x.tz_convert('Europe/Amsterdam'))
    agouti.timestamp = pd.to_datetime(agouti.timestamp)

    return agouti


def get_start_time_audio_file(audio_file_name: str) -> datetime:
    """Return start time of audio file as datetime object."""
    # get audio file date and time
    audio_file_date = audio_file_name.split('_')[3:5]
    audio_file_date = audio_file_date[0] + '_' + audio_file_date[1]
    # convert audio file date and time to datetime object
    audio_file_datetime = datetime.strptime(
        audio_file_date, '%Y-%m-%d_%H-%M-%S')
    # convert audio file datetime to UTC
    audio_file_datetime = audio_file_datetime.replace(tzinfo=pytz.UTC)
    return audio_file_datetime


def filter_agouti_on_audio_file(agouti: pd.DataFrame, audio_file_path: str) -> pd.DataFrame:
    """Return filtered agouti data for containing first two parts of audio file name and timestamps
    being between audio file start and end.

    Args:
        agouti (pd.DataFrame): main agouti dataframe
        audio_file_path (str): audio location

    Returns:
        pd.DataFrame: filtered agouti dataframe
    """

    audio_file_name = audio_file_path.split('\\')[-1]
    # get start time of audio file
    audio_file_datetime = get_start_time_audio_file(audio_file_name)
    # get duration of audio file
    with audioread.audio_open(audio_file_path) as f:
        totalsec = f.duration
    # filter agouti data for containing first two parts of audio file name
    filter_string = audio_file_name.split('_')[0:2]
    filter_string = filter_string[0] + '_' + filter_string[1]

    # further filter agouti data for timestamps being between audio file start and end
    begin = audio_file_datetime
    end = begin + pd.Timedelta(seconds=totalsec)

    agouti_filtered = agouti.query(
        'fileName.str.contains(@filter_string) & timestamp >= @begin and timestamp <= @end')
    return agouti_filtered


def load_and_filter_agouti_data(media_filepath: str, observations_filepath: str,
                                audio_file_path: str) -> pd.DataFrame:
    agouti = load_agouti_data(media_filepath, observations_filepath)
    agouti_filtered = filter_agouti_on_audio_file(agouti, audio_file_path)
    return agouti_filtered


def convert_filtered_data_into_list_with_start_and_end_times(
        agouti_filtered: pd.DataFrame, audio_file_path: str) -> list:
    """Convert filtered agouti data into list with difference between observation point and start time
    returns list of difference points in seconds."""
    # get start time of audio file
    audio_file_name = audio_file_path.split('\\')[-1]
    audio_file_datetime = get_start_time_audio_file(
        audio_file_name=audio_file_name)
    # for each instance in agouti filtered, see how far it is from the start of the audio file
    # and add the distance from the start to a list
    audio_points = []
    for _, row in agouti_filtered.iterrows():
        # get timestamp of instance
        timestamp = row['timestamp']
        # get distance from start of audio file
        distance_from_start = timestamp - audio_file_datetime
        # convert distance from start to seconds
        distance_from_start = distance_from_start.total_seconds()
        # add distance from start to list
        audio_points.append(int(distance_from_start))
    return audio_points


def create_list_with_start_and_end_times(audio_points: list, difference: int) -> list:
    """Create list with start time minus the difference and end time plus the difference.
    returns list with start and end times in seconds"""
    # create list with for each data point plus and minus difference
    audio_points_with_difference = []
    for point in audio_points:
        audio_points_with_difference.append(
            (point - difference, point + difference))
    return audio_points_with_difference


def create_list_with_random_start_and_end_times(audio_ranges: list, difference: int, audio_path: str) -> list:
    """Create list with random start time minus the difference and end time plus the difference.
    this list cannot contain any start or end times that are outside of the audio file or points within audio_points.
    returns list with start and end times in frames"""

    # get duration of audio file
    totalsec = get_audio_file_length(audio_path)
    # create list with random start and end times that are within the audio file and not within audio_points
    audio_points_with_difference = []

    for _ in range(len(audio_ranges)):
        # create random point between 0 and totalsec as integer
        random_point = random.randint(0, totalsec)
        # add difference to random point
        random_range = (random_point - difference, random_point + difference)
        # check if random range is within one of the ranges in audio_ranges or within audio_points_with_difference
        while any(random_range[0] <= r[1] and random_range[1] >= r[0] for r in
                  audio_ranges + audio_points_with_difference):
            # create new random point
            random_point = np.random.uniform(0, totalsec)
            # add difference to random point
            random_range = (random_point - difference,
                            random_point + difference)
        # add random range to list
        audio_points_with_difference.append(random_range)

    return audio_points_with_difference




