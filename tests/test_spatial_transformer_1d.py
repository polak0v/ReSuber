#!/usr/bin/python3
# coding: utf8
import os
import tensorflow as tf
import resuber.calculus as calculus

def test_spatial_transformer_1d():
    ### rigid
    # define input and target (shift: -1 and scale: 2)
    x = tf.constant([0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=tf.float32, name="x")
    target = tf.constant([0., 0., 0., 0., 0., 0.5, 1., 1., 1., 0.5, 0., 0.5, 1., 0.5, 0.], dtype=tf.float32, name="x")

    # compute output
    model = calculus.spatial_transformer_1d.SpatialTransformer1d()
    model.update_params(W=[0.5], b=[-1.])
    y = model(x)

    # verify
    tf.debugging.assert_near(target, y, atol=1e-9)

    ### non-rigid
    # define input and target
    x = tf.constant([0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.], dtype=tf.float32, name="x")
    target = tf.constant([0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.], dtype=tf.float32, name="x")

    # compute output
    model = calculus.spatial_transformer_1d.SpatialTransformer1d(rigid=False, mask=[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    model.update_params(B=[-2., -1., 1.])
    y = model(x)

    # verify
    tf.debugging.assert_near(target, y, atol=1e-9)