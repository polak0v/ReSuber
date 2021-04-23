import pysubs2
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

def read_audio(audio_filepath, target_fs=1000, mono=True):
    """Read an audio file and optionnaly resample it into a mono channel.

    Parameters
    ----------
        audio_filepath : string, required
            filepath to the audio WAV file
        target_fs : float
            target sampling rate (default: 1 kHz)
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
            sampling rate in Hz
    """
    wavfile.write(audio_filepath, fs, audio)

def read_subs(input_sub_filepath, target_fs=1000, encoding="utf-8"):
    """Read a subtitle file and convert it into a numerical signal.

    Parameters
    ----------
        input_sub_filepath : string, required
            filepath to the input subtitle SRT file
        target_fs : float
            target sampling rate in Hz (default: 1 kHz)
        encoding : string
            encoding to choose (default: utf-8)

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
    subs = pysubs2.load(input_sub_filepath, encoding=encoding)
    duration_sub_in_ms = subs[-1].end

    # box signal correspond to the subs
    target_ratio = target_fs / 1000.
    subs_signal = np.zeros(( np.int32(duration_sub_in_ms*target_ratio + 1),), dtype=np.float32)
    starts = np.array([], dtype=np.int32)
    ends = np.array([], dtype=np.int32)
    for sub in subs:
        start = np.int32(sub.start)
        start_ix = np.int32(sub.start*target_ratio)
        end = np.int32(sub.end)
        end_ix = np.int32(sub.end*target_ratio)
        starts = np.append(starts, start)
        ends = np.append(ends, end)
        subs_signal[start_ix:end_ix] = 1.

    return subs_signal, subs, starts, ends

def save_subs(subs, output_sub_filepath, encoding="utf-8"):
    """Save a subtitle file.

    Parameters
    ----------
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
        output_sub_filepath : string, required
            output filepath to the subtitle SRT file
        encoding : string
            encoding to choose (default: utf-8)
    """
    subs.save(output_sub_filepath, encoding=encoding)

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
    kernel = [norm * np.exp((-1)*(x**2)/(2 * sigma**2)) for x in range(-width, width + 1)]  

    return np.float32(kernel / np.sum(kernel))

def tf_1d_gaussian_filtering(signal, kernel_size):
    """Tensorflow convolution of the input signal with a gaussian kernel.

    Parameters
    ----------
        signal : `np.array` or `tf.tensor` [float], required
            input signal to be convolved with a gaussian kernel
        kernel_size : int. required
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
            input audio will be thresholded by this amount (default: np.quantile(audio, 1/4))
        kernel_size : int
            width of the 1D gaussian kernel for the input audio convolution (default: 100)
    
    Returns
    -------
        `np.array` [float] : input audio pre-processed
    """
    # get and convolve the audio magnitude spectrum
    audio_mag = np.abs(audio)
    audio_filtered = tf_1d_gaussian_filtering(audio_mag, kernel_size=kernel_size)
    default_threshold = np.quantile(audio_filtered[audio_mag>0], 1/4)
    # manual threshold to filter noise
    if threshold is None:
        threshold = default_threshold
    else:
        threshold = max(threshold, default_threshold)
    audio_preprocessed = np.array(audio_filtered > threshold, dtype=np.float32)

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
    duration_sub = subs_signal.shape[0]
    duration_audio = audio.shape[0]

    if duration_sub < duration_audio:
        duration = duration_audio - duration_sub
        subs_signal = tf.concat([subs_signal, tf.zeros((duration), dtype=tf.float32)], axis=0)
    else:
        duration = duration_sub - duration_audio
        audio = tf.concat([audio, tf.zeros((duration), dtype=tf.float32)], axis=0)

    return audio, subs_signal

def get_subs_mask(subs_signal, starts, ends, max_duration_allowed=10000, fs=1000):
    """Get a mask from a subtitle signal and starts/ends event, where each cluster is identified by its int value.

    Parameters
    ----------
        subs_signal : `np.array` [float], required
            input subtitle signal
        starts : `np.array` [np.int32], required
            subtitles starts events in ms
        ends : `np.array` [np.int32], required
            subtitles ends events in ms
        max_duration_allowed : int
            maximum duration in ms allowed between two subtitle events to form one cluster (default: 10000 ms)
        fs : float
            Sampling rate in Hz (default: 1 kHz)
    
    Returns
    -------
        `np.array` [float] : mask with cluster id values for each sample in the input subtitle signal
    """
    mask_subs_signal = np.zeros_like(subs_signal)
    num_sub = len(starts)
    middle_subs = np.array([], dtype=np.int32)
    fs_ratio = fs / 1000.
    # clustering the subs into n groups
    for i in range(num_sub - 1):
        dura = starts[i+1]*fs_ratio - ends[i]*fs_ratio
        if(dura > max_duration_allowed*fs_ratio):
            middle_sub = np.int32((ends[i]*fs_ratio + starts[i+1]*fs_ratio)/2)
            middle_subs = np.append(middle_subs, [middle_sub])
    mask_subs_signal[middle_subs] = 1

    return np.cumsum(mask_subs_signal)

def resync_subs(params, subs, mask=None, fs=1000, cut_start=0, cut_end=90060990):
    """Re-synchronize the subtitle object given the transformation parameters.

    Parameters
    ----------
        params : `list` [float], required
            transformation parameters
        subs : `pysubs2.ssafile.SSAFile`, required
            subtitle object with all the subtitle events
        mask : `np.array` [float]
            mask with cluster id values for each sample in the input subtitle signal
        fs : float
            Sampling rate in Hz (default: 1 kHz)
        cut_start : int
            Minimum timestamp value (ms) to include (default: 0)
        cut_end : int
            Maximum timestamp value (ms) to include (default: 90060990)
    
    Returns
    -------
        `pysubs2.ssafile.SSAFile` : subtitle object with all the re-synchronized subtitle events
    """
    target_ratio = fs / 1000.
    max_duration = int(subs[-1].end + 3600000)
    for sub, i in zip(subs, range(len(subs))):
        input_start = int(sub.start * target_ratio)
        input_end = int(sub.end * target_ratio)
        start =  params[0]*input_start + params[1]
        end = params[0]*input_end + params[1]
        # non-rigid
        if len(params) != 2:
            if mask is None:
                idx_start =  input_start
                idx_end = input_end
            else:
                idx_start = int(mask[input_start])
                idx_end = int(mask[input_end])
            start = start + params[2][idx_start]
            end = end + params[2][idx_end]
        if (cut_start < sub.start) & (sub.end < cut_end):
            # updating subtitle file
            subs[i].start = min(max(0, int(start/target_ratio)), max_duration)
            subs[i].end = min(max(0, int(end/target_ratio)), max_duration)

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

    start = int(subs[-1].end) + 2000
    end = start + 2000
    text = "Processed with <i>ReSuber</i>.\nCheck the github page <font color=\"blue\"> https://github.com/polak0v/ReSuber </font> !"
    event = pysubs2.SSAEvent(start=start, end=end, text=text)
    subs += [event]

    return subs