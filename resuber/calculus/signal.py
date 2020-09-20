import pysubs2
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

def read_audio(audio_filepath, target_fs=44100, mono=True):
    """Read an audio file and optionnaly resample it into a mono channel.

    Parameters
    ----------
        audio_filepath : string, required
            filepath to the audio WAV file
        target_fs : float
            target sampling rate (default: 44.1 kHz)
        mono : bool
            convert the audio into a mono-channel (default: True)

    Returns
    -------
        `np.array` [int16/int32/float32] : audio samples
    """
    fs, audio = wavfile.read(audio_filepath)
    channels = audio.shape[-1] if len(audio.shape) > 1 else 1

    # sampling the signal
    if target_fs !=  fs:
        n_samples = int(audio.shape[0]*(target_fs/fs))
        resample_idx = np.insert((fs/target_fs)*np.ones(n_samples - 1), 0, 0.)
        resample_idx = np.array(np.round(np.cumsum(resample_idx)), dtype=int)
        # getting mono channel
        audio_resampled = audio[resample_idx]
    else:
        audio_resampled = audio
    if mono & (channels > 1):
        audio_resampled = audio_resampled[:, 0]

    return audio_resampled, fs

def save_audio(audio, audio_filepath, fs):
    """Save an audio file.

    Parameters
    ----------
        audio : `np.array` [int16/int32/float32], required
            audio samples
        audio_filepath : string, required
            output filepath to the audio WAV file
        fs : float, required
            sampling rate
    """
    wavfile.write(audio_filepath, fs, audio)

def read_subs(input_sub_filepath):
    """Read a subtitle file and convert it into a numerical signal.

    Parameters
    ----------
        input_sub_filepath : string, required
            filepath to the input subtitle SRT file

    Returns
    -------
        subs_signal : `np.array` [np.int32] of size [last subtitle end event in ms + 1]
            subtitle signal (takes 0. or 1. value)
        subs : `pysubs2.ssafile.SSAFile`
            subtitle object with all the subtitle events
        starts : `np.array` [np.int32]
            subtitles starts events in ms
        ends : `np.array` [np.int32]
            subtitles ends events in ms
    """
    subs = pysubs2.load(input_sub_filepath, encoding="utf-8")
    duration_sub_in_ms = subs[-1].end

    # box signal correspond to the subs
    subs_signal = np.zeros((duration_sub_in_ms + 1,), dtype=np.int32)
    starts = np.array([], dtype=np.int32)
    ends = np.array([], dtype=np.int32)
    for sub in subs:
        start = np.int32(sub.start)
        end = np.int32(sub.end)
        starts = np.append(starts, start)
        ends = np.append(ends, end)
        subs_signal[start:end] = 1

    return subs_signal, subs, starts, ends

def save_subs(subs, output_sub_filepath):
    """Save a subtitle file.

    Parameters
    ----------
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
        output_sub_filepath : string, required
            output filepath to the subtitle SRT file
    """
    subs.save(output_sub_filepath)

def save_labels(subs, output_label_filepath):
    """Save a subtitle file into an audacity label TXT file.

    Parameters
    ----------
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
        output_label_filepath : string, required
            output filepath to the label TXT file
    """
    with open(output_label_filepath, "w") as fst:
        for sub in subs:
            start = sub.start
            end = sub.end
            text = sub.text
            fst.write("{}\t{}\t{}\n".format(start/1000., end/1000., text))
    fst.close()

def gaussian_kernel_1d(filter_length):
    """Get a 1D gaussian kernel.

    Parameters
    ----------
        filter_length : int, required
            width of the 1D gaussian kernel

    Returns
    -------
        `np.array` [float] of size [filter_length] : 1D gaussian kernel
    """
    #99% of the values
    sigma = (filter_length/2)/2.33
    width = int(filter_length/2)

    norm = 1.0 / (np.sqrt(2*np.pi) * sigma)
    kernel = [norm * np.exp((-1)*(x**2)/(2 * sigma**2)) for x in range(-width, width)]  

    return np.float32(kernel / np.sum(kernel))

def tf_1d_gaussian_filtering(signal, kernel_size=500):
    """Tensorflow convolution of the input signal with a gaussian kernel.

    Parameters
    ----------
        signal : `np.array` or `tf.tensor` [float], required
            input signal to be convolved with a gaussian kernel
        kernel_size : int
            width of the 1D gaussian kernel
    
    Returns
    -------
        `np.array` [float] : convolved input signal with a gaussian kernel
    """
    f = tf.reshape(gaussian_kernel_1d(kernel_size), [-1, 1, 1])
    signal_filtered = tf.reshape(tf.constant(signal, dtype=tf.float32), [1, -1, 1])

    return tf.reshape(tf.nn.conv1d(signal_filtered, filters=f, stride=1, padding='SAME'), [-1])

def filter_audio(audio, threshold=None, kernel_size=None):
    """Audio pre-processing for a vocal audio input (absolute, convolution and thresholding).

    Parameters
    ----------
        audio : `np.array` [float], required
            input audio signal
        threshold : float
            input audio will be thresholded by this amount (default: np.quantile(audio, 1/3))
        kernel_size : int
            width of the 1D gaussian kernel for the input audio convolution (default: 100)
    
    Returns
    -------
        `np.array` [float] : input audio pre-processed
    """
    # get and convolve the audio magnitude spectrum
    audio_mag = np.abs(audio)
    audio_filtered = tf_1d_gaussian_filtering(audio_mag, kernel_size=100)
    # manual threshold to filter noise
    if threshold is None:
        threshold = np.quantile(audio_filtered, 1/3)
    else:
        threshold = max(threshold, np.quantile(audio_filtered, 1/3))
    audio_preprocessed = np.array(audio_filtered > threshold, dtype=np.int32)

    return audio_preprocessed

def rescale_audio_subs(audio, subs_signal):
    """Resize both the input audio and subtitle signal so they have the same length.

    Parameters
    ----------
        audio : `np.array` [float], required
            input audio signal
        subs_signal : `np.array` [float], required
            input subtitle signal
    
    Returns
    -------
        audio : `np.array` [float]
            input audio signal resized
        subs_signal : `np.array` [float]
            input subtitle signal resized
    """
    # rescaling sub or audio so they have the same length
    duration_sub_in_ms = subs_signal.shape[0]
    duration_audio_in_ms = audio.shape[0]

    if duration_sub_in_ms < duration_audio_in_ms:
        duration = duration_audio_in_ms - duration_sub_in_ms
        subs_signal = tf.concat([subs_signal, tf.zeros((duration), dtype=tf.float32)], axis=0)
    else:
        duration = duration_sub_in_ms - duration_audio_in_ms
        audio = tf.concat([audio, tf.zeros((duration), dtype=tf.float32)], axis=0)

    return audio, subs_signal

def get_subs_mask(subs_signal, starts, ends, max_duration_allowed=10000):
    """Get a mask from a subtitle signal and starts/ends event, where each cluster is identified by its int value.

    Parameters
    ----------
        subs_signal : `np.array` [float], required
            input subtitle signal
        starts : `np.array` [float], required
            subtitles starts events in ms
        starts : `np.array` [float], required
            subtitles ends events in ms
        max_duration_allowed : int
            maximum duration allowed in ms between two subtitle events to form one cluster (default: 10000)
    
    Returns
    -------
        `np.array` [float] : mask with cluster id values for each sample in the input subtitle signal
    """
    mask_subs_signal = np.zeros_like(subs_signal)
    num_sub = len(starts)
    middle_subs = np.array([], dtype=np.int32)
    # clustering the subs into n groups
    for i in range(num_sub - 1):
        dura = np.int32(starts[i+1]) - np.int32(ends[i])
        if(dura > max_duration_allowed):
            middle_sub = np.int32((np.int32(ends[i]) + np.int32(starts[i+1]))/2)
            middle_subs = np.append(middle_subs, [middle_sub])
    mask_subs_signal[middle_subs] = 1

    return np.cumsum(mask_subs_signal)

def resync_subs(params, subs, max_duration_in_ms, mask=None):
    """Re-synchronize the subtitle object given the transformation parameters.

    Parameters
    ----------
        params : `list` [float], required
            transformation parameters
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
        max_duration_in_ms : float, required
            maximum time allowed of a subtitle event (should be the duration of the corresponding audio signal)
        mask : `np.array` [float]
            mask with cluster id values for each sample in the input subtitle signal
    
    Returns
    -------
        `pysubs2.ssafile.SSAFile` : subtitle object with all the re-synchronized subtitle events
    """
    for sub, i in zip(subs, range(len(subs))):
        start =  sub.start*params[0] + params[1]
        end = sub.end*params[0] + params[1]
        # non-rigid
        if len(params) != 2:
            if mask is None:
                idx_start = int(sub.start)
                idx_end = int(sub.end)
            else:
                idx_start = int(mask[int(sub.start)])
                idx_end = int(mask[int(sub.end)])
            start = start + params[2][idx_start]
            end = end + params[2][idx_end]
        # updating subtitle file
        subs[i].start = min(max(0, int(start)), max_duration_in_ms)
        subs[i].end = min(max(0, int(end)), max_duration_in_ms)

    return subs

def add_credits(subs):
    """Add credits to the software at the end of the subtitle SRT file.

    Parameters
    ----------
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
    
    Returns
    -------
        `pysubs2.ssafile.SSAFile` : subtitle object with credits appended at the end
    """
    subs[-1].text = subs[-1].text + "\n\n*** Re-synchronized with ReSuber, check the github page! ***"

    return subs

if __name__ == '__main__':
    audio_filepath = "examples/audio_example.wav"
    sub_filepath = "examples/audio_example.srt"
    label_filepath = "examples/audio_example.txt"

    # reading signals
    vocal, _ = read_audio(audio_filepath, target_fs=1000)
    subs_signal, subs, subs_starts, subs_ends = read_subs(sub_filepath)
    # preprocessing audio
    vocal_preprocessed = filter_audio(vocal, threshold=1000, kernel_size=100)
    # rescale to make sure the two signals have the same size
    vocal_preprocessed, subs_signal = rescale_audio_subs(vocal_preprocessed, subs_signal)
    # low-pass filtering to create dynamic for the signals (and avoid numerical instability during interpolation)
    subs_signal = tf_1d_gaussian_filtering(subs_signal, kernel_size=500)
    vocal_preprocessed = tf_1d_gaussian_filtering(vocal_preprocessed, kernel_size=500)
    # get mask
    mask = get_subs_mask(subs_signal, subs_starts, subs_ends, max_duration_allowed=100)
    # resyncs
    resynced_subs = resync_subs([1., 100.], subs, max_duration_in_ms=subs_signal.shape[0])
    # credits
    add_credits(subs)
    # save audacity label
    save_labels(subs, label_filepath)