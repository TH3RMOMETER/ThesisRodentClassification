import os
import time
import uuid
from datetime import timedelta

import audioread
import librosa
import numpy as np
import pandas as pd
import soundfile
import xarray as xr
from pydub import AudioSegment
from scipy import signal, stats
from skimage import util
from tqdm import tqdm

from config import Config


# open flac file from certain point in time

def create_list_with_audio_filepaths_in_folder(folder_path: str):
    """Create list with audio filepaths in folder

    Args:
        folder_path (str): path to folder with audio files

    Returns:
        List: list with audio filepaths
    """
    audio_filepaths = []
    # only include .flac files
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        audio_filepaths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.flac')]
    return audio_filepaths


def get_audio_file_length(file_path: str) -> int:
    """Returns length of audio file in seconds"""
    with audioread.audio_open(file_path) as f:
        totalsec = f.duration
    return totalsec


def get_sample_rate(file_path: str) -> int:
    """Returns sample rate of audio file"""
    return soundfile.SoundFile(file_path).samplerate


def convert_seconds_to_frames_in_audiofile(s: int, sample_rate: int) -> int:
    """Converts seconds to frames in audio file"""
    return round(s * sample_rate)


def open_flac_file(end_time, file_path: str, start_time: int = 0) -> tuple[np.ndarray, int]:
    """Opens flac file and returns audio and sample rate.
    start_time and end_time are in seconds."""
    # convert start_time and end_time to frames
    start_time = convert_seconds_to_frames_in_audiofile(
        start_time, get_sample_rate(file_path))
    end_time = convert_seconds_to_frames_in_audiofile(
        end_time, get_sample_rate(file_path))
    audio, sample_rate = soundfile.read(
        file_path, start=start_time, stop=end_time)
    return audio, sample_rate


def save_audio_file(audio: np.ndarray, sample_rate: int, output_path: str):
    """Saves audio file to file_path"""
    soundfile.write(output_path, audio, sample_rate)


def save_audio_file_start_end(
        output_path: str, start_time: int, end_time: int, file_path: str, index: int):
    """Saves audio file to file_path"""
    audio, sample_rate = open_flac_file(end_time, file_path, start_time)  # type: ignore
    soundfile.write(f'{output_path}{str(uuid.uuid4())}_{index}.flac', audio, sample_rate)


def create_slice_from_flac(
        file_path: str, slice_len: int,
        step_size: float, start_time: int, end_time) -> tuple[int, np.ndarray, int, float]:
    """Creates small slices from flac file. 
    slice_len is desired legth of each slice in ms. 
    step_size is how big of step to take between steps (larger size is less overlap)."""

    # read in flac file
    sig_data, sample_rate = open_flac_file(end_time, file_path, start_time)
    print('Sampling frequency: ' + str(sample_rate))

    # convert slice_len from ms to sample numbers
    slice_sample_num = slice_len * sample_rate / 1000

    # determine number of samples and length
    n_samples = sig_data.shape[0]
    print('Number of samples: ' + str(n_samples))
    sig_len = n_samples / sample_rate
    print('Length: ' + str(sig_len) + ' sec')

    # create slices
    steps = int(slice_sample_num * step_size)
    slices = util.view_as_windows(
        sig_data, window_shape=(slice_sample_num,), step=steps)
    print(f'Audio shape: {sig_data.shape}, Sliced audio shape: {slices.shape}')

    return sample_rate, slices, steps, sig_len


def create_spec_from_slice_array(slices: np.ndarray, steps: int,
                                 spec_window, NFFT, samp_freq) -> tuple[dict, np.ndarray, np.ndarray]:
    """Creates fft spectrogram from slice. spec_window is length of each segment (nperseg).
    NFFT is length of the FFT used (nfft). samp_freq is sampling frequency (in Hz) of slice (fs).
    steps is step size between slices"""

    spec_slices = {}
    samp_freq_kHz = samp_freq / 1000
    freqs_spec = np.ndarray
    times = np.ndarray

    for i in tqdm(range(slices.shape[0])):
        # spectrogram
        freqs_spec, times, Sx = signal.spectrogram(
            slices[i, :], fs=samp_freq, nperseg=spec_window, nfft=NFFT)

        time_stamp = (i * steps) / samp_freq_kHz

        # store as dic
        spec_slices[time_stamp] = Sx

    return spec_slices, freqs_spec, times  # type: ignore


def create_mfcc_from_slice_array(slices: np.ndarray, steps: int, spec_window, N_MFCC, samp_freq):
    """Creates mfcc from slice. spec_window is length of each segment.
    NFFT is length of the FFT used (nfft). samp_freq is sampling frequency (in Hz) of slice (fs).
    steps is step size between slices"""

    mfcc_slices = {}
    samp_freq_kHz = samp_freq / 1000
    freqs_mfcc, times, Sx = signal.spectrogram(
        slices[1, :], fs=samp_freq, nperseg=spec_window, nfft=N_MFCC)

    for i in tqdm(range(slices.shape[0])):
        mfcc = librosa.feature.mfcc(y=slices[i, :], sr=samp_freq)
        delta_Sx = librosa.feature.delta(mfcc)
        delta_delta_Sx = librosa.feature.delta(delta_Sx, order=2)

        time_stamp = (i * steps) / samp_freq_kHz

        # store as dic
        mfcc_slices[time_stamp] = [mfcc, delta_Sx, delta_delta_Sx]

    return mfcc_slices, freqs_mfcc, times  # type: ignore


def create_xarray_dataset_from_dic(dic: dict, freqs_spec: np.ndarray, times: np.ndarray) -> xr.DataArray:
    """Creates an xarray.Dataset object from a dictionary input."""

    slices_combined = {}

    for key, fft_slice in dic.items():
        slices_combined[key] = xr.DataArray(fft_slice, dims=('freq', 'times'), coords={
            'freq': freqs_spec, 'times': times})
    slices_Dataset = xr.Dataset(slices_combined).to_array(dim='slices')

    return slices_Dataset


def create_xarray_dataset_from_mfcc(dic: dict, freqs_spec: np.ndarray, times: np.ndarray) -> xr.DataArray:
    """Creates an xarray.Dataset object from a dictionary input."""

    slices_combined = {}

    for key, fft_slice in dic.items():
        slices_combined[key] = xr.DataArray(
            fft_slice, dims=('mfcc', 'mfcc_d', 'mfcc_d_d'))
    slices_Dataset = xr.Dataset(slices_combined).to_array(dim='slices')

    return slices_Dataset


# testing function


def get_remainders(slices_Dataset, slices, step_size, samp_freq, sig_len):
    slice_remainder = np.ceil(
        (sig_len - (slices.shape[0] * slices.shape[1] * step_size / samp_freq)) * 1000)
    netcdf_remainder = (sig_len * 1000) - \
                       slices_Dataset.slices.values[-1] - 22.5

    return slice_remainder, netcdf_remainder


def compute_spectral_features(Dataset: xr.DataArray) -> pd.DataFrame:
    """takes in netcdf dataset and computes 8 features and add to pandas dataframe"""

    spec_power = []
    spec_purs = []
    spec_centroid = []
    spec_spread = []
    spec_skewness = []
    spec_kurtosis = []
    spec_slope = []
    spec_roll_off = []

    freq_array = Dataset['freq'].values

    # compute power sum using groupby
    spec_power = Dataset.groupby('slices').sum(xr.ALL_DIMS)[
        '__xarray_dataarray_variable__'].values

    # compute other features for each slice individually
    for value in Dataset['slices'].values:
        spec_pur = stats.gmean(Dataset.sel(slices=value)['__xarray_dataarray_variable__'].values, axis=None) / \
                   Dataset.sel(  # type: ignore
                       slices=value)['__xarray_dataarray_variable__'].values.mean()

        mag_array = Dataset['__xarray_dataarray_variable__'].sel(
            slices=value).max(dim='times').values
        mag_probs = mag_array / sum(mag_array)
        freq_mag = freq_array * mag_probs

        spec_cent = sum(freq_mag)
        spec_spr = np.var(freq_mag)
        spec_skew = stats.skew(freq_mag)
        spec_kurt = stats.kurtosis(freq_mag)
        slope, _, _, _, _ = stats.linregress(
            freq_array, freq_mag)
        spec_ro = .95 * sum(freq_mag)

        spec_purs.append(spec_pur)
        spec_centroid.append(spec_cent)
        spec_spread.append(spec_spr)
        spec_skewness.append(spec_skew)
        spec_kurtosis.append(spec_kurt)
        spec_slope.append(slope)
        spec_roll_off.append(spec_ro)

    # add calculated lists to pandas dataframe
    spec_dataset = pd.DataFrame({'spec_power': spec_power, 'spec_purs': spec_purs, 'spec_centroid': spec_centroid,
                                 'spec_spread': spec_spread, 'spec_skewness': spec_skewness,
                                 'spec_kurtosis': spec_kurtosis, 'spec_slope': spec_slope,
                                 'spec_roll_off': spec_roll_off})

    return spec_dataset


def process_audio_NFFT(audio_path: str, slice_len: int, step_size: float,
                       start_time: int, output_path: str, end_time) -> None:
    """Process audio file into spectrograms and store in xarray Dataset.
    slice_len is desired legth of each slice in ms.
    step_size is how big of step to take between steps (larger size is less overlap)."""

    spec_window = 128
    NFFT = 512

    # process flac file of animal corresponding to annotations
    print(f'Start processing {audio_path.split("/")[-1]}')

    # create slices
    start = time.time()
    samp_freq, slices, steps, sig_len = create_slice_from_flac(
        audio_path, slice_len, step_size, start_time, end_time
    )
    end = time.time()
    print(str('Slices created in ' + str(end - start) + '  seconds'))

    # create spectrograms
    start = time.time()
    spec_slices, freqs_spec, times = create_spec_from_slice_array(
        slices, steps, spec_window, NFFT, samp_freq)
    end = time.time()
    print(str('Spectrograms created in ' + str(end - start) + '  seconds'))

    # create xarray Dataset
    start = time.time()
    slices_Dataset = create_xarray_dataset_from_dic(
        spec_slices, freqs_spec, times)
    end = time.time()
    print(str('xarray created in ' + str(end - start) + '  seconds'))

    # confirm timestamps are correct
    slice_remainder, netcdf_remainder = get_remainders(
        slices_Dataset, slices, step_size, samp_freq, sig_len)
    if slice_remainder != netcdf_remainder:
        raise Exception('Mismatch between slice and timestamp remainders')
    else:
        # save
        start = time.time()
        # calculate spectral features of slices_Dataset
        slices_Dataset = compute_spectral_features(slices_Dataset)
        end = time.time()
        print(str('xarray saved in ' + str(end - start) + '  seconds'))


def process_audio_MFCC(audio_path: str, slice_len: int, step_size: float,
                       start_time: int, output_path: str, index: int, end_time) -> None:
    """Process audio file into MFCC and store in xarray Dataset.
    slice_len is desired legth of each slice in ms.
    step_size is how big of step to take between steps (larger size is less overlap)."""

    spec_window = 128
    NFFT = 128

    # process flac file of animal corresponding to annotations
    print(f'Start processing {audio_path.split("/")[-1]}')

    # create slices
    start = time.time()
    samp_freq, slices, steps, sig_len = create_slice_from_flac(
        audio_path, slice_len, step_size, start_time, end_time
    )
    end = time.time()
    print(str('Slices created in ' + str(end - start) + '  seconds'))

    # create MFCCs
    start = time.time()
    spec_slices, freqs_spec, times = create_mfcc_from_slice_array(
        slices, steps, spec_window, NFFT, samp_freq)
    end = time.time()
    print(str('MFCCs created in ' + str(end - start) + '  seconds'))

    """ # create xarray Dataset
    start = time.time()
    slices_Dataset = create_xarray_dataset_from_mfcc(
        spec_slices, freqs_spec, times)
    end = time.time()
    print(str('xarray created in ' + str(end - start) + '  seconds')) """

    # create pandas dataframe with columns mfcc, mfcc_delta, mfcc_delta2
    slices_Dataset = pd.DataFrame.from_dict(spec_slices, orient='index', columns=[
        'mfcc', 'mfcc_delta', 'mfcc_delta2'])
    # convert dataset to pickle
    slices_Dataset.to_pickle(
        output_path + audio_path.split("\\")[-1].split(".")[0] + f'_{index}' '.pkl')


def get_timestamp_from_audio_file(audio_file_path: str):
    """Get timestamp from audio file

    Args:
        audio_file_path (str): audio file path

    Returns:
        str: timestamp
    """
    # split on \ and split on _ and take second to last and third to last element
    timestamp = audio_file_path.split("\\")[-1].split("_")[-3:-1]
    # join the two elements
    timestamp = "-".join(timestamp)
    timestamp = timestamp.split('-')
    # return timestamp as pd.Timestamp
    return pd.Timestamp(year=int(timestamp[0]), month=int(timestamp[1]), day=int(timestamp[2]), hour=int(timestamp[3]),
                        minute=int(timestamp[4]),
                        second=int(timestamp[5]), tz='UTC')


def create_pandas_dataframe_from_audio_files(list_with_audio_files: list) -> pd.DataFrame:
    """Create pandas dataframe from list with audio files

    Args:
        list_with_audio_files (list): list with audio files

    Returns:
        pd.DataFrame: pandas dataframe with audio files and timestamps
    """
    # create pandas dataframe with columns audio_file and timestamp
    audio_files = pd.DataFrame(list_with_audio_files, columns=['audio_files'])
    audio_files['timestamp'] = audio_files['audio_files'].apply(
        lambda x: get_timestamp_from_audio_file(x))
    # return dataframe
    return audio_files


def load_agouti_data(config: Config) -> pd.DataFrame:
    """Load agouti data

    Args:
        config (Config): Config object

    Returns:
        pd.DataFrame: agouti data
    """
    # load in agouti data
    agouti = pd.read_pickle(config.agouti_filepath)
    agouti = agouti.query(
        'fileName.str.contains("flevo") | fileName.str.contains("artis")')
    # dropna values for scientificName
    agouti = agouti.dropna(subset=['scientificName'])
    # filter on scientificName contains Rat
    agouti = agouti.query('scientificName.str.contains("Rat")')
    return agouti


def select_audio_files(config: Config, timestamp: pd.Timestamp, site_id: str,
                       audio_files: pd.DataFrame) -> pd.DataFrame:
    """
    Select audio files from audio folder that correspond to agouti annotations
    Args:
        audio_files: pd.DataFrame with audio files and timestamps
        site_id: str with site id
        timestamp: pd.Timestamp with timestamp
        config: Config object

    Returns: pd.DataFrame with audio files and timestamps

    """
    # create timestamps with plus and minus 2 minutes around timestamp
    start_time = timestamp - pd.Timedelta(minutes=2)
    end_time = timestamp + pd.Timedelta(minutes=2)
    # get all audio files with timestamp between start and end time
    audio_files = audio_files.query(
        'timestamp >= @start_time & timestamp <= @end_time')
    # sort on timestamp
    audio_files = audio_files.sort_values(by='timestamp')
    return audio_files


def merge_audio_files(audio_files: pd.DataFrame):
    """Merge audio files into one audio file

    Args:
        audio_files (pd.Dataframe): audio files

    Returns:
        pd.Dataframe: merged audio files
    """
    # get audio files
    audio_files = audio_files['audio_files'].tolist()
    # create audio file
    audio = AudioSegment.empty()
    # loop over audio files
    for audio_file in audio_files:
        # read audio file
        audio_file = AudioSegment.from_file(audio_file)
        # append audio file to audio
        audio = audio + audio_file
    # return audio
    return audio


def select_random_audio_point(existing_timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    Select a random timestamp within one hour of the existing timestamp, but not within ten minutes of the existing timestamp.
    Args:
        existing_timestamp: pd.Timestamp with existing timestamp

    Returns: pd.Timestamp with random timestamp
    """
    # Define the time range for the new timestamp
    start_time = existing_timestamp - timedelta(minutes=60)  # One hour before existing timestamp
    end_time = existing_timestamp + timedelta(minutes=60)  # One hour after existing timestamp

    # Generate a random timestamp within the defined range
    random_timestamp = pd.Timestamp(pd.to_datetime(
        start_time) + pd.DateOffset(seconds=np.random.randint(
        0, (end_time - start_time).total_seconds())))

    # Ensure the new timestamp is not within ten minutes of the existing timestamp
    while abs(existing_timestamp - random_timestamp) < timedelta(minutes=10):
        random_timestamp = pd.Timestamp(pd.to_datetime(
            start_time) + pd.DateOffset(seconds=np.random.randint(
            0, (end_time - start_time).total_seconds())))

    return random_timestamp


def create_audio_from_agouti_annotations(config: Config, agouti: pd.DataFrame, audio_files: pd.DataFrame):
    """Create audio from agouti annotations

    Args:
        config (Config): Config object
        agouti (pd.DataFrame): agouti annotations
        audio_files (pd.DataFrame): audio files
    """
    used_audio_files = set()
    # loop over agouti annotations with tqdm
    for index, row in tqdm(agouti.iterrows(), total=agouti.shape[0]):
        # get timestamp
        timestamp = row['timestamp']
        # get random timestamp with same tz as timestamp
        random_timestamp = select_random_audio_point(timestamp)

        # get site id from fileName
        site_id = row['fileName'].split('-')[1].split('_')[0:2]
        site_id = '_'.join(site_id)
        # select audio files if audio files are available
        selected_audio_files = select_audio_files(config, timestamp, site_id, audio_files)
        random_selected_audio_files = select_audio_files(config, random_timestamp, site_id, audio_files)
        # if dataframe is not empty
        if not audio_files.empty:
            # while random_audio_files is empty create new random timestamp and select new random audio files
            # ensure that the random audio files are not in the used audio files
            while random_selected_audio_files.empty or bool(
                    set(random_selected_audio_files['audio_files'].tolist()).intersection(
                        used_audio_files)):
                random_timestamp = select_random_audio_point(timestamp)
                random_selected_audio_files = select_audio_files(config, random_timestamp, site_id, audio_files)
            # add audio files values to used audio files set
            used_audio_files.update(random_selected_audio_files['audio_files'].tolist())
            used_audio_files.update(selected_audio_files['audio_files'].tolist())
            # merge audio files
            audio = merge_audio_files(selected_audio_files)
            random_audio = merge_audio_files(random_selected_audio_files)
            # export audio
            audio.export(config.output_path + '\\rats' + f'\\{site_id}_{timestamp.strftime("%Y_%m_%d_%H_%M")}.flac',
                         format='flac')
            random_audio.export(
                config.output_path + '\\noise' + f'\\{site_id}_{random_timestamp.strftime("%Y_%m_%d_%H_%M")}.flac',
                format='flac')


def test():
    config = Config()
    agouti = load_agouti_data(config)
    audio_files = create_list_with_audio_filepaths_in_folder(config.audio_folder)
    audio_files = create_pandas_dataframe_from_audio_files(audio_files)
    # create dummy pandas file with three timestamps and fileNames

    agouti = pd.DataFrame(columns=['timestamp', 'fileName'])
    agouti['timestamp'] = pd.to_datetime(
        ['2022-05-28 07:08:09+01:00', '2022-05-28 11:08:09+01:00', '2022-05-28 18:08:09+01:00'])
    agouti['fileName'] = ['20221209153156-artis_18_wildlifecamera1_2022-11-16_15-41-56_(136).JPG',
                          '20221209153156-artis_18_wildlifecamera1_2022-11-16_15-41-56_(136).JPG',
                          '20221209153156-artis_18_wildlifecamera1_2022-11-16_15-41-56_(136).JPG']
    create_audio_from_agouti_annotations(config, agouti, audio_files)
    x = 1


def run():
    config = Config()
    agouti = load_agouti_data(config)
    audio_files = create_list_with_audio_filepaths_in_folder(config.audio_folder)
    audio_files = create_pandas_dataframe_from_audio_files(audio_files)
    create_audio_from_agouti_annotations(config, agouti, audio_files)


if __name__ == '__main__':
    test()
