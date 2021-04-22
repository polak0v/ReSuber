import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import plotly.express as px
from .spatial_transformer_1d import SpatialTransformer1d

def plot_cost(cost, init_params, range_min, range_max, debug_dir):
    """Plot the cost image for the two parameters (weight and offset).

    Parameters
    ----------
        cost : `np.array` [`np.array` [float]], required
            cost matrix for the offset and weights
        init_params : `np.array` [float], required
            init weight and offset parameters
        range_min : `np.array` [float], required
            minimal range allowed for the weight and offset parameters, so top-left is [init_params[0] + range_min[0], init_params[1] + range_min[1]]
        range_max : `np.array` [float], required
            maximal range allowed for the weight and offset parameters, so right-bottom is [init_params[0] + range_max[0], init_params[1] + range_max[1]]
        debug_dir : string, required
            directory path where to store debugging information (cost image, iterations and model output)
    """
    # cost function over params
    fig = px.imshow(cost, color_continuous_scale='gray', aspect='auto', labels={'x':'offset (fs/1000 ms)', 'y':'ratio', 'color':'cost value (negative better)'},
                        x=np.linspace(init_params[1] + range_min[1], init_params[1] + range_max[1], cost.shape[1]),
                        y=np.linspace(init_params[0] + range_min[0], init_params[0] + range_max[0], cost.shape[0]) )
    fig.write_html('{}/cost_2d.html'.format(debug_dir))

def plot_data(model, x, target, debug_dir):
    """Plot the cost image for the two parameters (weight, offset and non-rigid offsets).

    Parameters
    ----------
        model : SpatialTransformer1d()
            spatial transformer model
        x : `tf.tensor` [float], required
            input signal
        target : `tf.tensor` [`tf.float32`], required
            target signal
        debug_dir : string, required
            directory path where to store debugging information (cost image, iterations and model output)
    """
    resamp_idx = int(6e4)
    y = model(x).numpy()[:resamp_idx]
    x = x.numpy()[:resamp_idx]
    t = target.numpy()[:resamp_idx]
    
    plotly_inputs = dict(input=x, model_output=y, target=t)
    plotly_outputs = ['input', 'model_output', 'target']
    if len(model.training_vars) > 2:
        if model.mask is None:
            B = model.B.numpy()
        else:
            B = tf.zeros(len(model.mask))
            for ii in len(model.B.numpy()):
                current_mask = tf.cast(tf.math.equal(model.mask, ii), dtype=tf.float32)
                B = B + current_mask * model.B.numpy()[ii]
        B = B[:resamp_idx]
        plotly_inputs["B"] = B
        plotly_outputs += ['offsets']
    
    # cost function over params
    fig = px.line(plotly_inputs, y=plotly_outputs, labels={'x':'time (ms)', 'y':'magnitude'})
    fig.write_html('{}/data.html'.format(debug_dir))

def plot_loss(losses, debug_dir):
    """Plot the loss function.

    Parameters
    ----------
        losses : `list` [`tf.tensor` [`tf.float32`]]
            loss for each  iteration
        debug_dir : string, required
            directory path where to store debugging information (cost image, iterations and model output)
    """
    # cost function over params
    fig = px.line(y=[losses[1:]], labels={'x':'iteration', 'y':'loss value'})
    fig.write_html('{}/loss.html'.format(debug_dir))

def loss(y, target, dv=False):
    """Negative correlation between the target and the output signal of the model.

    Parameters
    ----------
        y : `tf.tensor` [`tf.float32`], required
            output signal of the model
        target : `tf.tensor` [`tf.float32`], required
            target signal
        dv : bool
            compute the loss on the 1st order derivative of y and target (default: False)

    Returns
    -------
        float : loss value
    """
    if dv:
        y = y[1:] - y[:-1]
        target = target[1:] - target[:-1]
    loss = (-1) * tfp.stats.correlation(x=y, y=target, event_axis=None)
    return loss

def regularizer(model):
    """Diffusion regularizer on the spatial gradients of non-rigid parameters.

    Parameters
    ----------
        model : object, required
            model with `tensorflow` trainable parameters

    Returns
    -------
        float : regularizer value
    """
    return tf.reduce_mean((model.B[1:] - model.B[:-1])**2)

@tf.function
def train(model, x, target, l=0.):
    """Backpropagation step for the model.

    Parameters
    ----------
        model : object, required
            model with `tensorflow` trainable parameters
        x : `tf.tensor` [float], required
            input signal of the model
        target : `tf.tensor` [`tf.float32`], required
            target signal
        l : float
            regularizing parameter (default: 0.)

    Returns
    -------
        loss_value : float
            loss value for the current step
        grads : `list` [`tf.tensor`]
            gradients tensor for each trainable parameter of the model
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model(x), target, dv=False)
        if (l > 0) & (len(model.training_vars) > 2):
            loss_value = loss_value + l*regularizer(model)
    return loss_value, tape.gradient(loss_value, model.training_vars)

def rough_exploration(model, x, target, init_params, range_min, range_max, debug_dir=''):
    """Extract the optimal parameters with a rough exploration in the parameter (offset and weight) space.

    Parameters
    ----------
        model : object, required
            model with `tensorflow` trainable parameters
        x : `tf.tensor` [float], required
            input signal of the model
        target : `tf.tensor` [`tf.float32`], required
            target signal
        init_params : `np.array` [float], required
            init weight and offset parameters
        range_min : `np.array` [float], required
            minimal range allowed for the weight and offset parameters
        range_max : `np.array` [float], required
            maximal range allowed for the weight and offset parameters
        debug_dir : string
            directory path where to store debugging information (cost image, iterations and model output) (default: '')

    Returns
    -------
        `list` [float] : optimal transformation parameters (weight and offset)
    """
    weights = [np.linspace(init_params[0] + range_min[0], init_params[0] + range_max[0], 30)
            , np.linspace(init_params[1] + range_min[1], init_params[1] + range_max[1], 30)]
    cost = np.zeros((len(weights[0]), len(weights[1])), dtype=np.float32)

    # loop on all parameters combination
    for i in range(len(weights[0])):
        for j in range(len(weights[1])):
            model.update_params(W=[weights[0][i]], b=[weights[1][j]])
            cost[i, j] = loss(model(x), target)
    # updating model with best parameters
    argm = np.unravel_index(np.argmin(cost), cost.shape)
    params = [weights[0][argm[0]], weights[1][argm[1]]]
    if debug_dir:
        plot_cost(cost, init_params, range_min, range_max, debug_dir)

    return params

def fit(x, target, rigid=True, mask=None, max_offset_range=None, range_weight=[-1e-2, 1e-2], range_offset=[-5000., 5000.], w_trainable=True, b_trainable=True, fs=1000, debug_dir=''):
    """Fit a model with a rough exploration of the parameter space (from a linear model) followed by a gradient descent.

    Parameters
    ----------
        x : `tf.tensor` [float], required
            input signal
        target : `tf.tensor` [`tf.float32`], required
            target signal
        rigid : bool
            creates a lineal rigid model (default: true)
        mask : `np.array` [float] 
            mask with cluster id values for each sample in the input signal
        max_offset_range : float
            bound the offset trainable parameter by this value
        range_weight : `list` [float]
            range allowed for the weight parameter during rough exploration (default: [-1e-2, 1e-2])
        range_offset : `list` [float]
            range allowed in ms for the offset parameter during rough exploration (default: [-5000., 5000.])
        w_trainable : bool
            trainable weight or not (default: True)
        b_trainable : bool
            trainable bias or not (default: True)
        fs : float
            Sampling rate in Hz (default: 1 kHz)
        debug_dir : string
            directory path where to store debugging information (cost image, iterations and model output) (default: '')

    Returns
    -------
        `list` [float] : optimal transformation parameters (weight, offset and non-rigid bias for non-rigid model)
    """
    fs_ratio = (fs/1000.)
    # learning parameters (weight, offset, non-rigid bias) and optimizers
    lr_w = 1e-6
    lr_b = 1e0
    lr_B = lr_b * 1e2
    opt_w = tf.keras.optimizers.Adam(learning_rate=lr_w)
    opt_b = tf.keras.optimizers.Adam(learning_rate=lr_b)
    opt_B = tf.keras.optimizers.Adam(learning_rate=lr_B)
    # iterations parameters
    max_iters = 5000
    min_iters = int(0.05 * max_iters)
    min_loss_diff = 1e-9
    step = 0
    loss_diff = 1e9
    losses = [1e9]
    # regularizer parameter (for non-rigid)
    l = 10
    # number of cluster (masked non-rigid) or number of elements (fully non-rigid)
    num_clusters = tf.shape(x)[-1] if (mask is None) & (rigid == False) else None
    # Spatial transformer model
    model = SpatialTransformer1d(rigid=rigid, mask=mask, num_clusters=num_clusters, max_offset_range=max_offset_range * fs_ratio, w_trainable=w_trainable, b_trainable=b_trainable)

    # fitting with a rough exploration (linear model) follow by gradient descent
    rough_params = rough_exploration(model, x, target
                            , init_params=[1., 0.]
                            , range_min=[range_weight[0], range_offset[0] * fs_ratio]
                            , range_max=[range_weight[1], range_offset[1] * fs_ratio]
                            , debug_dir=debug_dir)
    model.update_params(W=[rough_params[0]], b=[rough_params[1]])
    if w_trainable | b_trainable:
        while ((loss_diff > min_loss_diff) | (step < min_iters)) & (step < max_iters):
            loss, grads = train(model, x, target, l)
            if debug_dir:
                if step % 10 == 0:
                    grads_no_none = []
                    for ii in range(len(grads)):
                        if grads[ii] is not None:
                            grad_is_nan = tf.math.reduce_any(tf.math.is_nan(grads[ii]))
                            if not grad_is_nan:
                                grads_no_none += [tf.reduce_mean(grads[ii])]
                    if rigid:
                        print("\t\tnon-rigid - Step: {} - loss: {} - Params: W {}, b {}".format(step, loss.numpy(), model.W.numpy(), model.b.numpy()))
                    else:
                        print("\t\tnon-rigid - Step: {} - loss: {} - Params: W {}, b {}, mean B {}".format(step
                                                                                                    , loss.numpy()
                                                                                                    , model.W.numpy()
                                                                                                    , model.b.numpy()
                                                                                                    , np.mean(model.B.numpy())))
                    print("\t\t\tGrads: {}, norm {}".format(np.mean(grads_no_none), tf.linalg.norm(tf.concat(grads_no_none, axis=0))))
            if w_trainable & b_trainable:
                opt_w.apply_gradients(zip(grads[:1], model.training_vars[:1]))
                opt_b.apply_gradients(zip(grads[1:2], model.training_vars[1:2]))
            if w_trainable & (not b_trainable):
                opt_w.apply_gradients(zip(grads[:1], model.training_vars[:1]))
            if (not w_trainable) & b_trainable:
                opt_b.apply_gradients(zip(grads[1:2], model.training_vars[1:2]))
            if not rigid:
                opt_B.apply_gradients(zip(grads[2:], model.training_vars[2:]))
            # end of iteration
            step = step + 1
            losses += [loss.numpy()]
            loss_diff = losses[step] - losses[step - 1]
            if loss_diff < 0 : loss_diff = 1e9
        if debug_dir:
            plot_loss(losses, debug_dir)
    # optimal parameters are the inverse of spatial transformer params
    if rigid:
        best_params = [1/model.training_vars[0].numpy(), (-1)*model.training_vars[1].numpy()]
    else:
        best_params = [1/model.training_vars[0].numpy(), (-1)*model.training_vars[1].numpy(), (-1)*model.training_vars[2].numpy()]
    if debug_dir:
        plot_data(model, x, target, debug_dir)
    return best_params