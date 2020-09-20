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
            output dir for the corrected SRT subtitle file (default: input_dir/resuber-output)
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
        refine_mode : {'mask', 'sample', 'no'}
            mask (cluster-wise), sample-wise or no non-linear refinement of the subtitles
        max_shift : float
            if non-linear refinement is allowed, define the maximum acceptable shift in ms
        min_clusters_distance : float
            if masked non-linear refinement is allowed, specify minimal distance allowed between clusters in ms
    """

    def __init__(self, debug=False, input_dir=None, output_dir=None, recursive=False, subtitles=None, vocals=None, refine='no', max_shift=500., min_clusters_distance=10000.):
        """Initialize the ReSuber class.

        Parameters
        ----------
            debug : bool
                enable debugging information (default: False)
            input_dir : string
                input directory to the SRT subtitle and WAV audio file(s) (default: .)
            output_dir : string 
                output dir for the corrected SRT subtitle file (default: input_dir/resuber-output)
            recursive : bool
                allow recursive search if vocals and/or subtitles are not specified (default: False)
            vocals : `list` [`string`]
                input filename(s) to the WAV vocal file(s) (default: ./*.wav)
            subtitles : `list` `list` [`string`]
                input filename(s) to the SRT subtitle file(s) per WAV vocal file (default: ./*.srt)
            refine : {'mask', 'sample', 'no'}
                mask (cluster-wise), sample-wise or no non-linear refinement of the subtitle signal (default: 'no')
            max_shift : float
                if non-linear refinement is allowed, define the maximum acceptable shift in ms (default: 500)
            min_clusters_distance : float
                if masked non-linear refinement is allowed, specify minimal distance allowed between clusters in ms (default: 10000)
        """
        self.debug = debug
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.recursive = recursive
        self.input_subtitle_names = subtitles
        self.vocal_names = vocals
        self.refine_mode = refine
        self.max_shift = max_shift
        self.min_clusters_distance = min_clusters_distance
        
        self.set_input_dir()
        self.set_output_dir()
        self.set_subtitles_and_vocals()
        if self.debug:
            self.set_debug_dir()

    def __repr__(self):
        return str(__file__) \
                + "\n" + str(datetime.datetime.now()) \
                + "\n" + str(platform.platform()) \
                + "\n" + "class ReSuber()" \
                + "\n\t debug : {}".format(self.debug) \
                + "\n\t input_dir : {}".format(self.input_dir) \
                + "\n\t output_dir : {}".format(self.output_dir) \
                + "\n\t recursive : {}".format(self.recursive) \
                + "\n\t subtitles : {}".format(self.input_subtitle_names) \
                + "\n\t vocals : {}".format(self.vocal_names) \
                + "\n\t refine : {}".format(self.refine_mode) \
                + "\n\t max_shift : {}".format(self.max_shift) \
                + "\n\t min_clusters_distance : {}".format(self.min_clusters_distance)

    def set_input_dir(self):
        """Set the input directory. """
        if self.input_dir is None:
            self.input_dir = os.getcwd()
        if not os.path.isdir(self.input_dir):
            raise ValueError('Directory {} does not exists!'.format(self.input_dir))

    def set_output_dir(self):
        """Set the output directory. """
        if self.output_dir is None:
            self.output_dir = os.path.join(self.input_dir, "resuber-output")
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
        all_subtitles = utils.dir.get_files_list(dir=self.input_dir, recursive=self.recursive, ext="srt")
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
            vocal, _ = calculus.signal.read_audio(vocal_filepath, target_fs=1000)
            vocal_preprocessed = calculus.signal.filter_audio(vocal, threshold=1000, kernel_size=100)
            # low-pass filtering to create dynamic for the binary signal (and avoid numerical instability during interpolation)
            vocal_preprocessed = calculus.signal.tf_1d_gaussian_filtering(vocal_preprocessed, kernel_size=500)

            for input_subtitle_filepath in self.input_subtitle_filepaths[i]:
                output_subtitle_filepath = os.path.join(self.output_dir, os.path.basename(input_subtitle_filepath[:-4] + "_resubed" + ".srt"))
                # reading subtitles signal
                subs_signal, subs, subs_starts, subs_ends = calculus.signal.read_subs(input_subtitle_filepath)
                # low-pass filtering to create dynamic the binary signal (and avoid numerical instability during interpolation)
                subs_signal = calculus.signal.tf_1d_gaussian_filtering(subs_signal, kernel_size=500)
                # rescale to make sure the two signals have the same size
                vocal_preprocessed, subs_signal = calculus.signal.rescale_audio_subs(vocal_preprocessed, subs_signal)
                mask_subs = None
                if self.refine_mode == "mask":
                    mask_subs = calculus.signal.get_subs_mask(subs_signal, subs_starts, subs_ends, max_duration_allowed=self.min_clusters_distance)
                # numerical optimization
                if self.debug:
                    debug_sub_dir = os.path.join(self.debug_dir, os.path.basename(input_subtitle_filepath[:-4]))
                best_params = calculus.ml.fit(subs_signal, vocal_preprocessed, 
                                                rigid=self.refine_mode == "no", mask=mask_subs, max_offset_range=self.max_shift, debug_dir=debug_sub_dir)
                # subtitle re-synchronization and save
                subs_resynced = calculus.signal.resync_subs(best_params, subs, mask=mask_subs, max_duration_in_ms=subs_signal.shape[0])
                calculus.signal.add_credits(subs_resynced)
                calculus.signal.save_subs(subs_resynced, output_subtitle_filepath)
                if self.debug:
                    output_label_filepath = os.path.join(self.debug_dir, os.path.basename(input_subtitle_filepath[:-4] + "_resubed" + ".txt"))
                    calculus.signal.save_labels(subs_resynced, output_label_filepath)
                
                print("\tsubtitle {} - params: {}".format(output_subtitle_filepath, best_params))
                #TODO target fs should also be used to reduce input subtitles data. (1 sample each ms is to much)