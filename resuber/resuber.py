import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import subprocess
import time
import datetime
import platform
import tensorflow as tf
import matplotlib.pyplot as plt

import resuber.utils as utils
import resuber.calculus as calculus

class ReSuber():
    """ Automatic tool to re-synchronize SRT subtitles given a vocal WAV file from a movie.
    
    Attributes
    ----------
        debug : bool
            enable debugging information (default: False)
        debug_dir : string
            directory path where to store debugging information during training
        input_dir : string
            input directory to the SRT subtitle and WAV audio file(s) (default: .)
        output_dir : string 
            output dir for the corrected SRT subtitle file (default: input_dir)
        debug_dir : string 
            debug dir for debugging files (default: output_dir/resuber-debug)
        recursive : bool
            allow recursive search if vocals and/or subtitles are not specified
        vocal_names : `list` [`string`]
            input filename(s) to the WAV vocal file(s) (default: ./*.wav)
        vocal_filepaths : `list` [`string`]
            input filepath(s) to the WAV vocal file(s) (input_dir + "/" + vocal_names)
        input_subtitle_names : `list` `list` [`string`]
            input filename(s) to the SRT subtitle file(s) per WAV vocal file
        input_subtitle_filepaths : `list` `list` [`string`]
            input filepaths(s) to the SRT subtitle file(s) per WAV vocal file (input_dir + "/" + input_subtitle_names)
        encoding : `string`
            encoding for subtitles (default: utf-8)
        fs : float
            sampling rate in Hz
        start_ms : int
            Minimum timestamp (ms) to process (default: 0)
        end_ms : int
            Maximum timestamp (ms) to process (default: 90060990)
        start_idx : int
            Minimum timestamp (sample) to process (default: 0)
        end_idx : int
            Maximum timestamp (sample) to process (default: 90060990 * fs/1000)
        range_weight : `list` [float]
            range allowed for the weight parameter during rough exploration
        range_offset : `list` [float]
            range allowed in ms for the offset parameter during rough exploration
        w_trainable : bool
            enable the optimization of the weight parameter (default: True)
        b_trainable : bool
            enable the optimization of the offset parameter (default: True)
        refine_mode : {'mask', 'sample', 'no'}
            mask (cluster-wise), sample-wise or no non-linear refinement of the subtitles
        max_shift : float
            if non-linear refinement is allowed, define the maximum acceptable shift in ms
        min_clusters_distance : float
            if masked non-linear refinement is allowed, specify minimal distance allowed between clusters in ms
    """

    def __init__(self
                , debug=False
                , input_dir=None
                , output_dir=None
                , recursive=False
                , subtitles=None
                , vocals=None
                , encoding="utf-8"
                , fs=100
                , start='0:0:0.0'
                , end='24:60:60.99'
                , range_weight=[-1e-2, 1e-2]
                , range_offset=[-5000., 5000.]
                , fix_weight=False
                , fix_offset=False
                , refine='no'
                , max_shift=500.
                , min_clusters_distance=10000.):
        """Initialize the ReSuber class.

        Parameters
        ----------
            debug : bool
                enable debugging information (default: False)
            input_dir : string
                input directory to the SRT subtitle and WAV audio file(s) (default: .)
            output_dir : string 
                output dir for the corrected SRT subtitle file (default: input_dir)
            recursive : bool
                allow recursive search if vocals and/or subtitles are not specified (default: False)
            vocals : `list` [`string`]
                input filename(s) to the WAV vocal file(s) (default: ./*.wav)
            encoding : `string`
                encoding for subtitles (default: utf-8)
            subtitles : `list` `list` [`string`]
                input filename(s) to the SRT subtitle file(s) per WAV vocal file (default: ./*.srt)
            fs : float
                sampling rate in Hz (default: 100 Hz)
            start : string
                Minimum timestamp to process with format 'h:m:s.ms' (default: '0:0:0.0')
            end : string
                Maximum timestamp to process with format 'h:m:s.ms (default: '24:60:60.99')
            range_weight : `list` [float]
                range allowed for the weight parameter during rough exploration (default: [-1e-2, 1e-2])
            range_offset : `list` [float]
                range allowed in ms for the offset parameter during rough exploration (default: [-5000., 5000.])
            fix_weight : bool
                disable the optimization of the weight parameter (default: False)
            fix_offset : bool
                disable the optimization of the offset parameter (default: False)
            refine : {'mask', 'sample', 'no'}
                mask (cluster-wise), sample-wise or no non-linear refinement of the subtitle signal (default: 'no')
            max_shift : float
                if non-linear refinement is allowed, define the maximum acceptable shift in ms (default: 500 ms)
            min_clusters_distance : float
                if masked non-linear refinement is allowed, specify minimal distance allowed between clusters in ms (default: 10000 ms)
        """
        self.debug = debug
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.recursive = recursive
        self.input_subtitle_names = subtitles
        self.vocal_names = vocals
        self.encoding = encoding
        self.fs = fs
        self.start = start
        self.end = end
        self.range_weight = range_weight
        self.range_offset = range_offset
        self.w_trainable = not fix_weight
        self.b_trainable = not fix_offset
        self.refine_mode = refine
        self.max_shift = max_shift
        self.min_clusters_distance = min_clusters_distance
        
        self.set_start_end_ms()
        self.set_input_dir()
        self.set_output_dir()
        self.set_subtitles_and_vocals()
        if self.debug:
            self.set_debug_dir()

    def __repr__(self):
        return str(__file__) \
                + "\n" + str(datetime.datetime.now()) \
                + "\n" + str(platform.platform()) \
                + "\n" + "class {} - {}".format(self.__class__.__name__, utils.args.get_version()) \
                + "\n\t debug : {}".format(self.debug) \
                + "\n\t input_dir : {}".format(self.input_dir) \
                + "\n\t output_dir : {}".format(self.output_dir) \
                + "\n\t recursive : {}".format(self.recursive) \
                + "\n\t subtitles : {}".format(self.input_subtitle_names) \
                + "\n\t vocals : {}".format(self.vocal_names) \
                + "\n\t encoding : {}".format(self.encoding) \
                + "\n\t fs : {}".format(self.fs) \
                + "\n\t start : {}".format(self.start) \
                + "\n\t end : {}".format(self.end) \
                + "\n\t range-weight : {}".format(self.range_weight) \
                + "\n\t range-offset : {}".format(self.range_offset) \
                + "\n\t fix-weight : {}".format(not self.w_trainable) \
                + "\n\t fix-offset : {}".format(not self.b_trainable) \
                + "\n\t refine : {}".format(self.refine_mode) \
                + "\n\t max-shift : {}".format(self.max_shift) \
                + "\n\t min-clusters_distance : {}".format(self.min_clusters_distance)

    def set_input_dir(self):
        """Set the input directory. """
        if self.input_dir is None:
            self.input_dir = os.getcwd()
        if not os.path.isdir(self.input_dir):
            raise ValueError('Directory {} does not exists!'.format(self.input_dir))

    def set_output_dir(self):
        """Set the output directory. """
        if self.output_dir is None:
            self.output_dir = self.input_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def set_debug_dir(self):
        """Set the debug directory. """
        self.debug_dir = os.path.join(self.output_dir, "resuber-debug")
        for input_subtitle_filepaths in self.input_subtitle_filepaths:
            for input_subtitle_filepath in input_subtitle_filepaths:
                debug_sub_dir = os.path.join(self.debug_dir, os.path.basename(input_subtitle_filepath[:-4]))
                if not os.path.isdir(debug_sub_dir):
                    os.makedirs(debug_sub_dir)

    def set_start_end_ms(self):
        """Utility to convert from date string format into int (ms)"""
        time_start_list = self.start.split(":")
        time_end_list = self.end.split(":")
        start_ms = int(float(time_start_list[0])*60*60*1000 + float(time_start_list[1])*60*1000 + float(time_start_list[2])*1000)
        end_ms = int(float(time_end_list[0])*60*60*1000 + float(time_end_list[1])*60*1000 + float(time_end_list[2])*1000)
        if start_ms < 0 | start_ms > 90060990:
            start_ms = 0
        if end_ms < 0 | end_ms > 90060990:
            end_ms = 90060990
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.start_idx = int(self.start_ms * (self.fs/1000))
        self.end_idx = int(self.end_ms * (self.fs/1000))

    def subtitle_group(self):
        """Group all the subtitle file paths by their name. """
        subtitles_filepaths_flat = [item for sublist in self.input_subtitle_filepaths for item in sublist]
        self.input_subtitle_filepaths = []
        for subtitle_filepath in sorted(subtitles_filepaths_flat, key=len):
            # if the subtitle filepath was already injected, we ignore it
            if not subtitle_filepath in [item for sublist in self.input_subtitle_filepaths for item in sublist]:
                self.input_subtitle_filepaths += [utils.dir.filepath_match(subtitles_filepaths_flat, subtitle_filepath)]

    def init_subtitles_and_vocals(self):
        """Initialize the subtitle and vocal file paths. """
        self.input_subtitle_filepaths = None
        self.vocal_filepaths = None

        if self.input_subtitle_names is not None:
            self.input_subtitle_filepaths = [[os.path.join(self.input_dir, subtitle_name) for subtitle_name in input_subtitle_names] for input_subtitle_names in self.input_subtitle_names]
        if self.vocal_names is not None:
            self.vocal_filepaths = [os.path.join(self.input_dir, vocal_name) for vocal_name in self.vocal_names]

    def set_subtitles_and_vocals(self):
        """Set the subtitle and vocal file paths. """
        self.init_subtitles_and_vocals()
        all_subtitles = utils.dir.get_files_list(dir=self.input_dir, recursive=self.recursive, ext="srt", exclude="resubed")
        all_vocals = utils.dir.get_files_list(dir=self.input_dir, recursive=self.recursive, ext="wav")

        # if subtitles are empty, vocals are used to match and get subtitles
        if self.input_subtitle_filepaths is None:
            self.input_subtitle_filepaths = []
            if self.vocal_filepaths is None:
                self.vocal_filepaths = all_vocals
            for vocal_filepath in self.vocal_filepaths:
                if not os.path.isfile(vocal_filepath):
                    raise ValueError('{} does not exists!'.format(vocal_filepath))
                if vocal_filepath.split('.')[-1] != "wav":
                    raise ValueError('Input {} is not in WAV format!'.format(vocal_filepath))
                if not vocal_filepath in [item for sublist in self.input_subtitle_filepaths for item in sublist]:
                    if utils.dir.filepath_match(all_subtitles, vocal_filepath):
                        self.input_subtitle_filepaths += [utils.dir.filepath_match(all_subtitles, vocal_filepath)]
        # otherwise, subtitles are used to match and get vocals (if empty)
        else:
            if self.vocal_filepaths is not None:
                if len(self.vocal_filepaths) != len(self.input_subtitle_filepaths):
                    self.subtitle_group()
            else:
                if len(all_vocals) != len(self.input_subtitle_filepaths):
                    self.subtitle_group()
                self.vocal_filepaths = []
                for subtitle_filepaths in self.input_subtitle_filepaths:
                    for sub_filepath in subtitle_filepaths:
                        if not os.path.isfile(sub_filepath):
                            raise ValueError('{} does not exists!'.format(sub_filepath))
                        if sub_filepath.split('.')[-1] != "srt":
                            raise ValueError('{} not in SubRip format!'.format(sub_filepath))
                        # subtitles with country code in the form ".COUNTRY_CODE.srt" needs a supplementary check
                        if not utils.dir.filepath_match(all_vocals, sub_filepath):
                            vocal_filepath = utils.dir.filepath_match(all_vocals, '.'.join(sub_filepath.split('.')[:-1]))
                        else:
                            vocal_filepath = utils.dir.filepath_match(all_vocals, sub_filepath)
                            break
                    if vocal_filepath:
                        self.vocal_filepaths += vocal_filepath
        
        if len(self.vocal_filepaths) != len(self.input_subtitle_filepaths):
            raise ValueError('Number of vocal files ({}) does not match number of subtitle groups ({}) ! \n {} \n {}'.format(
                                len(self.vocal_filepaths), len(self.input_subtitle_filepaths), self.vocal_filepaths, self.input_subtitle_filepaths))

    def __call__(self):
        print(self.__repr__() + "\n")
        start_time = time.time()
        # initialize tensorflow
        if self.debug:
            tf.config.list_physical_devices('GPU')
        tf.debugging.set_log_device_placement(False)
        tf.nn.conv1d(tf.ones((1, 1, 1)), filters=tf.ones((1, 1, 1)), stride=1, padding='SAME')
        # get number of vocal audio file
        n_vocals = len(self.vocal_filepaths)
        for i in range(n_vocals):
            vocal_filepath = self.vocal_filepaths[i]
            print("Processing {}/{}".format(i+1, n_vocals))
            print("\tvocal {}".format(vocal_filepath))
            # reading and pre-processing vocal
            vocal, _ = calculus.signal.read_audio(vocal_filepath, target_fs=self.fs)
            vocal = tf.constant(vocal, dtype=tf.float32)
            ksize_audio = max(10, int(2000. * (self.fs / 1000.)))
            vocal_preprocessed = calculus.signal.filter_audio(vocal, kernel_size=ksize_audio)
            # low-pass filtering to create dynamic for the binary signal (and avoid numerical instability during interpolation)
            ksize = max(10, int(500. * (self.fs / 1000.)))
            vocal_preprocessed = calculus.signal.tf_1d_gaussian_filtering(vocal_preprocessed, kernel_size=ksize)

            for input_subtitle_filepath in self.input_subtitle_filepaths[i]:
                output_subtitle_filepath = os.path.join(self.output_dir, os.path.basename(input_subtitle_filepath + ".resubed.srt"))
                print("\tsubtitle {}".format(output_subtitle_filepath))
                # reading subtitles signal
                subs_signal, subs, subs_starts, subs_ends = calculus.signal.read_subs(input_subtitle_filepath, target_fs=self.fs, encoding=self.encoding)
                subs_signal = tf.constant(subs_signal, dtype=tf.float32)
                # low-pass filtering to create dynamic the binary signal (and avoid numerical instability during interpolation)
                subs_signal = calculus.signal.tf_1d_gaussian_filtering(subs_signal, kernel_size=ksize)
                # rescale to make sure the two signals have the same size
                vocal_preprocessed, subs_signal = calculus.signal.rescale_audio_subs(vocal_preprocessed, subs_signal)
                mask_subs = None
                if self.refine_mode == "mask":
                    mask_subs = calculus.signal.get_subs_mask(subs_signal, subs_starts, subs_ends, max_duration_allowed=self.min_clusters_distance, fs=self.fs)
                    mask_subs = mask_subs[self.start_idx:self.end_idx]
                # numerical optimization
                debug_sub_dir = ""
                if self.debug:
                    debug_sub_dir = os.path.join(self.debug_dir, os.path.basename(input_subtitle_filepath[:-4]))
                # start and stop idx, in number of samples
                vocal_preprocessed = vocal_preprocessed[self.start_idx:self.end_idx]
                subs_signal = subs_signal[self.start_idx:self.end_idx]
                # gradient descent fitting
                best_params = calculus.ml.fit(subs_signal
                                            , vocal_preprocessed
                                            , rigid=self.refine_mode == "no"
                                            , mask=mask_subs
                                            , max_offset_range=self.max_shift
                                            , range_weight=self.range_weight
                                            , range_offset=self.range_offset
                                            , w_trainable=self.w_trainable
                                            , b_trainable=self.b_trainable
                                            , fs=self.fs
                                            , debug_dir=debug_sub_dir)
                # subtitle re-synchronization and save
                subs_resynced = calculus.signal.resync_subs(best_params
                                                            , subs
                                                            , mask=mask_subs
                                                            , fs=self.fs
                                                            , cut_start=self.start_ms
                                                            , cut_end=self.end_ms)
                calculus.signal.add_credits(subs_resynced)
                calculus.signal.save_subs(subs_resynced, output_subtitle_filepath, encoding="utf-8")
                if self.debug:
                    output_label_filepath = os.path.join(self.debug_dir, os.path.basename(input_subtitle_filepath[:-4] + "_resubed" + ".txt"))
                    calculus.signal.save_labels(subs_resynced, output_label_filepath)
                
                print("\tparams: {}".format(best_params))
        print("###")
        print("ReSuber duration {} s".format(time.time() - start_time))