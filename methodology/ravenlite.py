import time
from datetime import datetime

import audioread
import numpy as np
import pandas as pd
import pytz
import soundfile
import xarray as xr
from scipy import signal
from skimage import util
from microdict import mdict
import polars as pl


def load_agouti_data(media_filepath, observations_filepath):
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
    agouti.timestamp = pd.to_datetime(agouti.timestamp)

    return agouti


def filter_agouti_on_audio_file(agouti: pd.DataFrame, audio_file_path: str):
    # split audio file path into parts
    audio_file_path_parts = audio_file_path.split('\\')
    # get audio file name
    audio_file_name = audio_file_path_parts[-1]
    # get audio file date and time
    audio_file_date = audio_file_name.split('_')[3:5]
    audio_file_date = audio_file_date[0] + '_' + audio_file_date[1]
    # convert audio file date and time to datetime object
    audio_file_datetime = datetime.strptime(
        audio_file_date, '%Y-%m-%d_%H-%M-%S')
    # convert audio file datetime to UTC
    audio_file_datetime = audio_file_datetime.replace(tzinfo=pytz.UTC)
    # get duration of audio file
    with audioread.audio_open(audio_file_path) as f:
        totalsec = f.duration

    # filter agouti data for containing first two parts of audio file name
    filter_string = audio_file_name.split('_')[0:2]
    filter_string = filter_string[0] + '_' + filter_string[1]
    agouti_filtered = agouti.query('fileName.str.contains(@filter_string)')

    # further filter agouti data for timestamps being between audio file start and end
    begin = audio_file_datetime
    end = begin + pd.Timedelta(seconds=totalsec)
    agouti_filtered = agouti_filtered.query(
        'timestamp >= @begin and timestamp <= @end')
    return agouti_filtered


def create_slice_from_flac(file_path, file_len, slice_len, step_size):
    """Creates small slices from flac file. file_length is desired length of file in minutes.
    slice_len is desired legth of each slice in ms. 
    step_size is how big of step to take between steps (larger size is less overlap)."""

    # read in flac file
    sig_data, samp_freq = soundfile.read(file_path)
    print('Sampling frequency: ' + str(samp_freq))

    # if file_length is None, use entire file
    if file_len is None:
        file_len = sig_data.shape[0]/samp_freq/60
        file_len = file_len.__floor__()
    # convert file_length from min to sample numbers
    file_length_num = file_len*samp_freq*60

    # convert slice_len from ms to sample numbers
    slice_sample_num = slice_len*samp_freq/1000

    # use only slices within file_length
    sig_data = sig_data[0:file_length_num.__floor__()]

    # determine number of samples and length
    n_samples = sig_data.shape[0]
    print('Number of samples: ' + str(n_samples))
    sig_len = n_samples/samp_freq
    print('Length: ' + str(sig_len) + ' sec')

    # create slices
    steps = int(slice_sample_num*step_size)
    slices = util.view_as_windows(
        sig_data, window_shape=(slice_sample_num,), step=steps)
    print(f'Audio shape: {sig_data.shape}, Sliced audio shape: {slices.shape}')

    return samp_freq, slices, steps, sig_len


def create_xarray_dataset_from_dic(dic, freqs_spec, times):
    """Creates an xarray.Dataset object from a dictionary input."""

    slices_combined = {}

    for key, fft_slice in dic.items():
        slices_combined[key] = xr.DataArray(fft_slice, dims=('freq', 'times'), coords={
                                            'freq': freqs_spec, 'times': times})
    slices_Dataset = xr.Dataset(slices_combined).to_array(dim='slices')

    return slices_Dataset


def create_xarray_dataset_from_soundfile_blocks(file_path, slice_len, overlap, NFFT, spec_window):
    """Creates xarray.Dataset object from soundfile blocks. slice_len is length of each slice in ms.
    overlap is the overlap between slices in ms."""

    flac_file = soundfile.SoundFile(file_path)
    samp_freq = flac_file.samplerate
    print('Sampling frequency: ' + str(samp_freq))

    # Create block generator for flac file
    block_gen = flac_file.blocks(blocksize=int(samp_freq*slice_len/1000),
                                 overlap=int(samp_freq*overlap/1000),
                                 fill_value=0,
                                 frames=-1)

    # For each block in generator, create fft spectrogram and add to xarray.Dataset
    slices_combined = {}

    def block_func():
        for i, block in enumerate(block_gen):
            if i % 5000 == 0:
                print(i)
            # spectrogram
            freqs_spec, times, Sx = signal.spectrogram(
                block, fs=samp_freq, nperseg=spec_window, nfft=NFFT)
            yield Sx
    blockgen = block_func()
    slices_combined = np.fromiter(
        blockgen, dtype=np.dtype(('float64', (NFFT//2+1, 55))))
    x = 1
    """ 
    for i, block in enumerate(block_gen):
        if i % 5000 == 0:
            print(i)
        # spectrogram
        freqs_spec, times, Sx = signal.spectrogram(
            block, fs=samp_freq, nperseg=spec_window, nfft=NFFT)
        # store as dic
        slices_combined[i] = Sx
 """
    # convert to xarray.Dataset
    # slices_Dataset = create_xarray_dataset_from_dic(
    #   slices_combined, freqs_spec, times)

    # return slices_Dataset, freqs_spec, times


def create_spec_from_slice_array(slices, steps, spec_window, NFFT, samp_freq):
    """Creates fft spectrogram from slice. spec_window is length of each segment (nperseg).
    NFFT is length of the FFT used (nfft). samp_freq is sampling frequency (in Hz) of slice (fs).
    steps is step size between slices"""

    spec_slices = {}
    samp_freq_kHz = samp_freq/1000

    for i in range(slices.shape[0]):
        if i % 5000 == 0:
            print(i)

        # spectrogram
        freqs_spec, times, Sx = signal.spectrogram(
            slices[i, :], fs=samp_freq, nperseg=spec_window, nfft=NFFT)

        time_stamp = (i*steps) / samp_freq_kHz

        # store as dic
        spec_slices[time_stamp] = Sx

    return spec_slices, freqs_spec, times


# testing function
def get_remainders(slices_Dataset, slices, step_size, samp_freq, sig_len):
    slice_remainder = np.ceil(
        (sig_len - (slices.shape[0] * slices.shape[1] * step_size / samp_freq))*1000)
    netcdf_remainder = (sig_len * 1000) - \
        slices_Dataset.slices.values[-1] - 22.5

    return slice_remainder, netcdf_remainder


def convert_to_ravenlite(audio_file_name: str, agouti: pd.DataFrame) -> pd.DataFrame:
    filter_string = audio_file_name.split('\\')[-1].split('_')
    filter_string = f'{filter_string[0]}_{filter_string[1]}_wildlife wildlife camera1_{filter_string[3]}'
    agouti_filtered = agouti.query('fileName.str.contains(@filter_string)')

    with audioread.audio_open(audio_file_name) as f:
        totalsec = f.duration
    time_start = audio_file_name.split('_')[-2]
    timestamp = agouti_filtered.timestamp.iloc[0]
    begin = datetime.combine(
        timestamp.date(), datetime.strptime(time_start, '%H-%M-%S').time())
    end = begin + pd.Timedelta(seconds=totalsec)
    # filter filtered agouti data such that timestamps are between begin and end
    agouti_filtered = agouti_filtered.query(
        'timestamp >= @begin and timestamp <= @end')


def run():
    file_len = None  # full length
    slice_len = 25
    step_size = 0.9

    spec_window = 128
    NFFT = 512
    path = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark_1_audio1_2021-09-28_16-00-00_(0).flac'

    # process flac file of animal corresponding to annotations
    print(f'Processing {path}')

    # create slices
    start = time.time()
    samp_freq, slices, steps, sig_len = create_slice_from_flac(
        path, file_len, slice_len, step_size)
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
        slices_Dataset.to_netcdf('flevo' + '_xr_Dataset.nc')
        end = time.time()
        print(str('xarray saved in ' + str(end - start) + '  seconds'))


def run2():
    file_len = None  # full length
    slice_len = 25
    step_size = 0.9
    overlap = 0.5

    spec_window = 128
    NFFT = 512
    path = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark_1_audio1_2021-09-28_16-00-00_(0).flac'

    # process flac file
    print(f'Processing {path}')

    # create xarray Dataset
    start = time.time()
    slices_Dataset = create_xarray_dataset_from_soundfile_blocks(
        path, slice_len, overlap, NFFT, spec_window)
    end = time.time()
    print(str('xarray created in ' + str(end - start) + '  seconds'))

    # save
    start = time.time()
    slices_Dataset.to_netcdf('flevo' + '_xr_Dataset.nc')
    end = time.time()
    print(str('xarray saved in ' + str(end - start) + '  seconds'))


def run3():
    audiopath = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark_1_audio1_2021-09-28_16-00-00_(0).flac'
    # load in the data from flevo
    agouti_media = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\media.csv'
    agouti_observations = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\observations.csv'
    agouti = load_agouti_data(agouti_media, agouti_observations)

    # filter on audio file
    agouti_filtered = filter_agouti_on_audio_file(agouti, audiopath)
    x=1

run3()
