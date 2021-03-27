#!/usr/bin/python3
# coding: utf8
import os
import resuber.calculus as calculus

def test_signal():
    audio_filepath =  os.path.join(os.path.dirname(__file__), "..", "examples", "audio_example.wav")
    sub_filepath =  os.path.join(os.path.dirname(__file__), "..", "examples", "audio_example.fr.srt")
    label_filepath =  os.path.join(os.path.dirname(__file__), "..", "examples", "audio_example.fr.txt")

    # reading signals
    fs = 100
    vocal, _ = calculus.signal.read_audio(audio_filepath, target_fs=fs)
    subs_signal, subs, subs_starts, subs_ends = calculus.signal.read_subs(sub_filepath, target_fs=fs)
    # preprocessing audio
    vocal_preprocessed = calculus.signal.filter_audio(vocal, threshold=1000, kernel_size=10)
    # rescale to make sure the two signals have the same size
    vocal_preprocessed, subs_signal = calculus.signal.rescale_audio_subs(vocal_preprocessed, subs_signal)
    # low-pass filtering to create dynamic for the signals (and avoid numerical instability during interpolation)
    subs_signal = calculus.signal.tf_1d_gaussian_filtering(subs_signal, kernel_size=50)
    vocal_preprocessed = calculus.signal.tf_1d_gaussian_filtering(vocal_preprocessed, kernel_size=50)
    # get mask
    mask = calculus.signal.get_subs_mask(subs_signal, subs_starts, subs_ends, max_duration_allowed=100, fs=fs)
    # resyncs
    resynced_subs = calculus.signal.resync_subs([1., 100.*(fs/1000.)], subs, fs=fs)
    # credits
    calculus.signal.add_credits(subs)
    # save audacity label
    calculus.signal.save_labels(subs, label_filepath)