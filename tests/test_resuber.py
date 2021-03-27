#!/usr/bin/python3
# coding: utf8
import os
import resuber

def test_resuber():
    input_dir =  os.path.join(os.path.dirname(__file__), "..", "examples")
    resuber.ReSuber(input_dir=input_dir)()