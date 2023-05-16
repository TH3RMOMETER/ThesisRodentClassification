from load_agouti import (load_and_filter_agouti_data, convert_filtered_data_into_list_with_start_and_end_times,
                         create_list_with_start_and_end_times, create_list_with_random_start_and_end_times)
from process_audio import process_audio_NFFT, process_audio_MFCC


def run():
    # load in agouti data
    audiopath = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark_1_audio1_2021-09-28_16-00-00_(0).flac'

    agouti_media = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\media.csv'
    agouti_observations = r'C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\observations.csv'

    output_path = r'C:\Users\gijst\Documents\Master Data Science\Thesis\processed_data'

    slice_len = 25
    step_size = 0.9

    agouti = load_and_filter_agouti_data(
        media_filepath=agouti_media, observations_filepath=agouti_observations, audio_file_path=audiopath)
    audio_points = convert_filtered_data_into_list_with_start_and_end_times(
        agouti_filtered=agouti, audio_file_path=audiopath)
    audio_points = create_list_with_start_and_end_times(
        audio_points=audio_points, difference=120)
    random_audio_points = create_list_with_random_start_and_end_times(
        audio_ranges=audio_points, difference=120, audio_path=audiopath)
    # for each point in audio_points, process audio
    """ for index, point in enumerate(audio_points):
        process_audio_MFCC(
            audio_path=audiopath, start_time=point[0], end_time=point[1], slice_len=slice_len, step_size=step_size, output_path=f'{output_path}\\rats\\', index=index) """
    print(random_audio_points)
    for index, point in enumerate(random_audio_points):
        process_audio_MFCC(
            audio_path=audiopath, start_time=point[0], end_time=point[1], slice_len=slice_len, step_size=step_size, output_path=f'{output_path}\\noise\\', index=index)

if __name__ == "__main__":

    run()
