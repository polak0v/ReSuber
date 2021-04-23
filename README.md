

# ReSuber

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/polak0v/ReSuber/HEAD) [![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PyPI version](https://badge.fury.io/py/resuber.svg)](https://badge.fury.io/py/resuber)  ![GitHub](https://img.shields.io/github/license/polak0v/resuber) [![Donate with Bitcoin](https://en.cryptobadges.io/badge/small/12eAEKU4rgvhLCxvdkxKJYocJdEFRyNrta)](https://en.cryptobadges.io/donate/12eAEKU4rgvhLCxvdkxKJYocJdEFRyNrta)



![](logo.svg)

ReSuber is an automatic tool to re-synchronize any SRT subtitles, using its corresponding vocal audio WAV stream from a movie.

It uses machine learning techniques to perform language agnostic re-synchronization, by checking the signal correlation between the vocal audio stream and the corrected subtitle signal.

ReSuber also provide different utilities, including vocal audio extraction, subtitle/video file merging into containers and automatic translation of SRT subtitles with the Google Cloud Translation API (no API key required).

[License](LICENSE)

# Usage

The main tool of this toolbox, `resuber`, re-synchronize any `.srt` subtitle file, given its corresponding `.wav` audio vocal file from a movie.
For the complete list of tools, you should read [this paragraph](#optionnal-tools).

## First use

Given any directory with the following file-structure (and same name):

```
  |-examples
  |  |-movie_example.fr.srt
  |  |-movie_example.eng.srt
  |  |-movie_example.wav
```
Simply run `resuber` from within the directory:

```
cd examples
resuber
```
After completion, each subtitle will be re-synchronized and saved into new files `*.resubed.srt`:

```
  |-examples
  |  |-movie_example.fr.srt
  |  |-movie_example.eng.srt
  |  |-movie_example.wav
  |  |-movie_example.fr.srt.resubed.srt
  |  |-movie_example.eng.srt.resubed.srt
```
If you don't have the audio track for your movie, follow [this section](#i-dont-have-the-audio-track-from-my-movie).

Input subtitle language extensions must follow [ISO_639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

## Advanced use

### Arguments

```
usage: resuber [-h] [--debug] [--input-dir INPUT_DIR]
               [--output-dir OUTPUT_DIR] [--recursive]
               [--subtitles [SUBTITLES [SUBTITLES ...]]]
               [--vocals [VOCALS [VOCALS ...]]] [--refine {no,mask,sample}]
               [--fs FS] [--start START] [--end END]
               [--range-weight [RANGE_WEIGHT [RANGE_WEIGHT ...]]]
               [--range-offset [RANGE_OFFSET [RANGE_OFFSET ...]]]
               [--fix-weight] [--fix-offset] [--max-shift MAX_SHIFT]
               [--min-clusters-distance MIN_CLUSTERS_DISTANCE]
               [--encoding ENCODING] [--version]

optional arguments:
  -h, --help            show this help message and exit
  --debug               enable debugging information (default: False)
  --input-dir INPUT_DIR
                        input directory to the SRT subtitle and WAV audio
                        file(s) (default: .)
  --output-dir OUTPUT_DIR
                        output dir for the corrected SRT subtitle file
                        (default: input_dir)
  --recursive           allow recursive search if vocals and/or subtitles are
                        not specified (default: False)
  --subtitles [SUBTITLES [SUBTITLES ...]]
                        input filename(s) to the SRT subtitle file(s) per WAV
                        vocal file (default: ./*.srt)
  --vocals [VOCALS [VOCALS ...]]
                        input filename(s) to the WAV vocal file(s) (default:
                        ./*.wav)
  --refine {no,mask,sample}
                        mask (cluster-wise), sample-wise or no non-linear
                        refinement of the subtitle signal (default: 'no')
  --fs FS               sampling rate in Hz (default: 100 Hz)
  --start START         Minimum timestamp to process with format 'h:m:s.ms'
                        (default: '0:0:0.0')
  --end END             Maximum timestamp to process with format 'h:m:s.ms
                        (default: '24:60:60.99')
  --range-weight [RANGE_WEIGHT [RANGE_WEIGHT ...]]
                        range allowed for the weight parameter during rough
                        exploration (default: [-1e-2, 1e-2])
  --range-offset [RANGE_OFFSET [RANGE_OFFSET ...]]
                        range allowed in ms for the offset parameter during
                        rough exploration (default: [-5000., 5000.])
  --fix-weight          disable the optimization of the weight parameter
                        (default: False)
  --fix-offset          disable the optimization of the offset parameter
                        (default: False)
  --max-shift MAX_SHIFT
                        if non-linear refinement is allowed, define the
                        maximum acceptable shift in ms (default: 500 ms)
  --min-clusters-distance MIN_CLUSTERS_DISTANCE
                        if masked non-linear refinement is allowed, specify
                        minimal distance allowed between clusters in ms
                        (default: 10000 ms)
  --encoding ENCODING   encoding for subtitles (default: utf-8)
  --version             show program's version number and exit
```

### Customize the AI

`resuber` makes the assumption that the subtitles are "linearly" de-synchronized with the audio (i.e. it exists a function such as y = α x + β). This is typically the case for subtitles that were created from a movie with a different framerate.

There are several parameters that allows you to have some control over the fitting algorithm, more specifically the scaling α and offset parameter β. 
1. You can first change the range for the initial parameter search with `--range-weight` and `--range-offset`. This can be usefull if you know approximately how your inputs are de-synchronized.
2. It is also possible to disable the gradient-descent loop with `--fix-weight` and `--fix-offset`. In this case, the algorithm simply make a standard search in the parameter space (defined by `--range-*`).
3. Though I would recommend to use [pysubs2](https://github.com/tkarabela/pysubs2) for that, you can also edit the subtitle yourself and completely disable the AI. For example if you know that your subtitle have a fixed delay of +200ms, use:
    ```
    resuber --range-weight -0.0001 0.0001 --range-offset -200.1 -199.9 --fix-weight --fix-offset
    ```
### Debugging

I advice you to enable the `--debug` argument, it will create a new folder `resuber-debug` in which you will find usefull files.

```
  |-resuber-debug
  |  |-movie_example.fr
  |  |  |-data.html
  |  |  |-loss.html
  |  |  |-cost_2d.html
  |  |-movie_example.eng_resubed.txt
  |  |-movie_example.eng
  |  |  |-data.html
  |  |  |-loss.html
  |  |  |-cost_2d.html
  |  |-movie_example.fr_resubed.txt
```
`*_resubed.txt` is the processed subtitle converted into a `.txt` [Audacity](https://www.audacityteam.org/) label file. This is really usefull if you want to load the audio track, and view the subtitles.

`*.[lang]` folders contains intermediate and output from the algorithm:
* `cost_2d.html` correspond to the initial affine parameters (scale and offset) before the gradient-descent. Ideally, the shape of the cost should be gaussian-like (if) linear de-synchronization).
* `data.html` is a truncated view of the input subtitle signal (blue), target audio (green) and re-synchronized subtitle signal (orange). Re-synchronized subtitle signal should be as much close as possible to the target audio.
* `loss.html` show the variation of the fitness (correlation) at each iteration. Ideally, it should be an asymptotic decreasing function.

### Non-linear refinement (WIP)

Where `resuber` works relatively well for linear de-synchronization, it still has trouble for non-linear de-synchronization.

If that is the case, you can try to use the `--refine` argument. With `mask`, `resuber` will decompose the signal into multiple clusters, and fit the best delay for each cluster (typically the case for anime song opening with varying lengths). `sample` means that each value will have a different delay.

# Optionnal tools

ReSuber is a software with different tools, each serving its own purpose.
Instead of creating a big pipeline (and messing up with dependencies), I made multiple tools that are independent between each other.
To know which tool is best suited for your case, select one of the following section.

[>> I don't have the audio track from my movie](#i-dont-have-the-audio-track-from-my-movie)

[>> I have an original (correct) subtitle, and I want to synchronize another one](#i-have-an-original-correct-subtitle-and-i-want-to-synchronize-another-one)

[>> I want to translate my subtitle](#i-want-to-translate-my-subtitle)

[>> I have a subtitle that I want to merge with my movie file](#i-have-a-subtitle-that-i-want-to-merge-with-my-movie-file)

## I don't have the audio track from my movie

[Spleeter](https://github.com/deezer/spleeter) from [Deezer](https://www.deezer.com/us/) is a deep-learning tool that separates vocal track from an audio track.
This is obviously really usefull here, since the music and sound effects "pollutes" the signal and makes the correlation between the subtitle and audio signal a mess.

I made a tool called `spleeter2resuber` that:
1. take the audio track from the `.mp4`, `.mkv` or `.avi` movie file
2. extract the (human) vocal from the audio with `spleeter`.

After installing `spleeter` and [ffmpeg](https://www.ffmpeg.org/), just run `spleeter2resuber` inside any folder with:

```
  |-examples
  |  |-movie_example.mkv
  |  |-movie_example.fr
```
It will create a new folder `resuber`, ready to be use. Check also the sub-folder `resuber/spleeter` for debuging.

## I have an original (correct) subtitle, and I want to synchronize another one

If that is the case and you were not successfull with `resuber` (for example, the other subtitle is non-linearly dependent on the audio) then you have another solution.

You can use `resuber-move` that will take the reference timestamp from the (correct) subtitle, and do a nearest neighbor with your other (de-synchronized) subtitle.
That way you will have the text from the (de-synchronized) subtitle, with approximately the reference timestamp of the (correct) subtitle.

Given this folder:
```
  |-examples
  |  |-movie_example.fr.srt (original)
  |  |-movie_example.eng.srt (de-synchronized)
  |  |-movie_example.wav
```
Run:
```
resuber-move --ref-lang en --tgt-time fr
```

You have several other options to play with.

```
usage: resuber-move [-h] [--input-dir INPUT_DIR]
                    [--ref-lang REFERENCE_LANGUAGE]
                    [--tgt-time TARGET_TIMESTAMPS] [--encoding ENCODING]
                    [--min-dist MIN_DIST] [--drop-far DROP_FAR]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
  --ref-lang REFERENCE_LANGUAGE
  --tgt-time TARGET_TIMESTAMPS
  --encoding ENCODING
  --min-dist MIN_DIST   Maximum delay (ms) to be considered as a near neighbor
                        (default: 2000ms).
  --drop-far DROP_FAR   Drop subtitles that are too far.
```

## I want to translate my subtitle

Hopefully, if you are lazy like me there is the [google translate API](https://translate.google.com).

With these files:
```
  |-examples
  |  |-movie_example.fr.srt
  |  |-movie_example.wav
```

Translate from french subtitle `movie_example.fr.srt` to english `movie_example.eng.srt` with:

```
resuber-translate --ref-lang fr --tgt-lang en
```
Google imposes several limitations with the number of requests. I managed to process the translation batch by batch to reduce the burden, but be careful to not run this too often (or use a VPN...).

```
usage: resuber-translate [-h] [--input-dir INPUT_DIR]
                         [--ref-lang REFERENCE_LANGUAGE]
                         [--tgt-lang TARGET_LANGUAGE] [--encoding ENCODING]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
  --ref-lang REFERENCE_LANGUAGE
  --tgt-lang TARGET_LANGUAGE
  --encoding ENCODING
```

## I have a subtitle that I want to merge with my movie file

For this you can use [mkvmerge](https://mkvtoolnix.download/doc/mkvmerge.html), but I also made a home-made tool `resuber-merge` that requires [ffmpeg](https://www.ffmpeg.org/) to be installed.

Given a folder with a movie file (`.mp4`, `.mkv` or `.avi`) and subtitle:
```
  |-examples
  |  |-movie_example.fr.srt
  |  |-movie_example.mp4
```

You can merge the subtitle inside a video `.mkv` file with:
```
resuber-merge
```

Make sure to check all the options.

```
usage: resuber-merge [-h] [--input-dir INPUT_DIR]
                     [--output_container OUTPUT_CONTAINER]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
  --output_container OUTPUT_CONTAINER
```

# Installation

## pip

```
python3 -m pip install resuber
```

## src

```
git clone https://github.com/polak0v/ReSuber
cd ReSuber
make install
```

# Unit testing

You will need [pytest](https://docs.pytest.org/en/6.2.x/), and then run:

```
pytests tests/
```

# Support

If you like my work, you are welcome to donate some bitcoins. :)

BTC: 12eAEKU4rgvhLCxvdkxKJYocJdEFRyNrta