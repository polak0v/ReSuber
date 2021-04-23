#!/usr/bin/python3
# coding: utf8
import os
import resuber
import tensorflow as tf

def test_resuber():
    input_dir =  os.path.join(os.path.dirname(__file__), "..", "examples")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    resuber.ReSuber(input_dir=input_dir, output_dir=output_dir)()

    target_sub = os.path.join(output_dir, "audio_example.{lang}.srt.resubed.srt")
    ref_sub = os.path.join(input_dir, "audio_example.{lang}.srt.resubed.srt")
    # checking french and english output
    for lang in ["fr", "eng"]:
        target_signal = resuber.calculus.signal.read_subs(target_sub.format(lang=lang))[0]
        ref_signal = resuber.calculus.signal.read_subs(ref_sub.format(lang=lang))[0]
        # rescaling both subtitile signal (naming for the function not optimal)
        target_signal, ref_signal = resuber.calculus.signal.rescale_audio_subs(target_signal, ref_signal)
        # comparison
        tf.debugging.assert_near(ref_signal, target_signal, atol=1e-3)

if __name__ == "__main__":
    test_resuber()
