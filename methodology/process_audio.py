import time
from sys import getsizeof

import audioread
import numpy as np
import pandas as pd
import polars as pl
import soundfile
import xarray as xr
from scipy import signal, stats
from skimage import util
from tqdm import tqdm
import librosa

# open flac file from certain point in time


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
    return s*sample_rate


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
    # print length of audio file in seconds from start_time to end_time
    print('Length of audio file: ' + str(audio.shape[0]/sample_rate) + ' sec')
    return audio, sample_rate


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
    slice_sample_num = slice_len*sample_rate/1000

    # determine number of samples and length
    n_samples = sig_data.shape[0]
    print('Number of samples: ' + str(n_samples))
    sig_len = n_samples/sample_rate
    print('Length: ' + str(sig_len) + ' sec')

    # create slices
    steps = int(slice_sample_num*step_size)
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
    samp_freq_kHz = samp_freq/1000
    freqs_spec = np.ndarray
    times = np.ndarray

    for i in tqdm(range(slices.shape[0])):
        # spectrogram
        freqs_spec, times, Sx = signal.spectrogram(
            slices[i, :], fs=samp_freq, nperseg=spec_window, nfft=NFFT)

        time_stamp = (i*steps) / samp_freq_kHz

        # store as dic
        spec_slices[time_stamp] = Sx

    return spec_slices, freqs_spec, times  # type: ignore


def create_mfcc_from_slice_array(slices: np.ndarray, steps: int, spec_window, N_MFCC, samp_freq):
    """Creates mfcc from slice. spec_window is length of each segment.
    NFFT is length of the FFT used (nfft). samp_freq is sampling frequency (in Hz) of slice (fs).
    steps is step size between slices"""

    mfcc_slices = {}
    samp_freq_kHz = samp_freq/1000
    freqs_mfcc, times, Sx = signal.spectrogram(
        slices[1, :], fs=samp_freq, nperseg=spec_window, nfft=N_MFCC)

    for i in tqdm(range(slices.shape[0])):
        mfcc = librosa.feature.mfcc(y=slices[i, :], sr=samp_freq)
        delta_Sx = librosa.feature.delta(mfcc)
        delta_delta_Sx = librosa.feature.delta(delta_Sx, order=2)

        time_stamp = (i*steps) / samp_freq_kHz

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
        (sig_len - (slices.shape[0] * slices.shape[1] * step_size / samp_freq))*1000)
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

        spec_pur = stats.gmean(Dataset.sel(slices=value)['__xarray_dataarray_variable__'].values, axis=None) / Dataset.sel(  # type: ignore
            slices=value)['__xarray_dataarray_variable__'].values.mean()

        mag_array = Dataset['__xarray_dataarray_variable__'].sel(
            slices=value).max(dim='times').values
        mag_probs = mag_array/sum(mag_array)
        freq_mag = freq_array*mag_probs

        spec_cent = sum(freq_mag)
        spec_spr = np.var(freq_mag)
        spec_skew = stats.skew(freq_mag)
        spec_kurt = stats.kurtosis(freq_mag)
        slope, _, _, _, _ = stats.linregress(
            freq_array, freq_mag)
        spec_ro = .95*sum(freq_mag)

        spec_purs.append(spec_pur)
        spec_centroid.append(spec_cent)
        spec_spread.append(spec_spr)
        spec_skewness.append(spec_skew)
        spec_kurtosis.append(spec_kurt)
        spec_slope.append(slope)
        spec_roll_off.append(spec_ro)

    # add calculated lists to pandas dataframe
    spec_dataset = pd.DataFrame({'spec_power': spec_power, 'spec_purs': spec_purs, 'spec_centroid': spec_centroid,
                                 'spec_spread': spec_spread, 'spec_skewness': spec_skewness, 'spec_kurtosis': spec_kurtosis, 'spec_slope': spec_slope, 'spec_roll_off': spec_roll_off})

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
    slices_Dataset = pd.DataFrame.from_dict(spec_slices, orient='index', columns=['mfcc', 'mfcc_delta', 'mfcc_delta2'])
    # convert dataset to pickle
    slices_Dataset.to_pickle(output_path + audio_path.split("\\")[-1].split(".")[0] + f'_{index}' '.pkl')
