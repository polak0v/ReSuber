import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from .spatial_transformer_1d import SpatialTransformer1d

def loss(y, target, dv=False):
    """Absolute of one substracted by the correlation between the target and the output signal of the model.

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
    loss = tf.abs(1. - tfp.stats.correlation(x=y, y=target, event_axis=None))
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

def plot_cost(cost, init_params, range_min, range_max, debug_dir):
    """Plot the cost image for the two parameters (weight and offset).

    Parameters
    ----------
        cost : `np.array` [`np.array` [float]], required
            cost matrix for the offset and weights
        init_params : `np.array` [float], required
            init weight and offset parameters
        range_min : `np.array` [float], required
            minimal range allowed for the weight and offset parameters, so the top-left of the image is [params[0] + range_min[0], params[1] + range_min[1]]
        range_max : `np.array` [float], required
            maximal range allowed for the weight and offset parameters, so the right-bottom of the image is [params[0] + range_max[0], params[1] + range_max[1]]
        debug_dir : string, required
            directory path where to store debugging information (cost image, iterations and model output)
    """
    # cost function over params
    plt.figure()
    plt.imshow(cost[::-1, :], extent=[init_params[1] + range_min[1]
                            , init_params[1] + range_max[1]
                            , init_params[0] + range_min[0]
                            , init_params[0] + range_max[0]], aspect="auto", cmap="gray")
    plt.xlabel("offset")
    plt.ylabel("ratio")
    plt.savefig('{}/cost_image.png'.format(debug_dir))

def plot_data(model, x, target, debug_dir):
    """Plot the cost image for the two parameters (weight and offset).

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
    # cost function over params
    plt.figure()
    plt.plot(target, "k", label='target')
    plt.plot(x, "b--", label='input')
    plt.plot(model(x), "g--", label='model output')
    plt.legend()
    plt.savefig('{}/data.png'.format(debug_dir))

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
        loss_value = loss(model(x), target)
        if l > 0.:
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
            minimal range allowed for the weight and offset parameters, so the top-left of the image is [init_params[0] + range_min[0], init_params[1] + range_min[1]]
        range_max : `np.array` [float], required
            maximal range allowed for the weight and offset parameters, so the right-bottom of the image is [init_params[0] + range_max[0], init_params[1] + range_max[1]]
        debug_dir : string
            directory path where to store debugging information (cost image, iterations and model output) (default: '')

    Returns
    -------
        model : object, required
            optimized model with `tensorflow` trainable parameters
    """
    weights = [np.linspace(init_params[0] + range_min[0], init_params[0] + range_max[0], 20)
            , np.linspace(init_params[1] + range_min[1], init_params[1] + range_max[1], 20)]
    cost = np.zeros((len(weights[0]), len(weights[1])), dtype=np.float32)

    # loop on all parameters combination
    for i in range(len(weights[0])):
        for j in range(len(weights[1])):
            model.update_params(W=[weights[0][i]], b=[weights[1][j]])
            cost[i, j] = loss(model(x), target)
    # updating model with best parameters
    argm = np.unravel_index(np.argmin(cost), cost.shape)
    params = [weights[0][argm[0]], weights[1][argm[1]]]
    model.update_params(W=[params[0]], b=[params[1]])
    if debug_dir:
        plot_cost(cost, init_params, range_min, range_max, debug_dir)

    return model

def fit(x, target, rigid=True, mask=None, max_offset_range=None, debug_dir=''):
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
        debug_dir : string
            directory path where to store debugging information (cost image, iterations and model output) (default: '')

    Returns
    -------
        `list` [float] : optimal transformation parameters (weight, offset and non-rigid bias for non-rigid model)
    """
    # learning parameters (weight, offset, non-rigid bias) and optimizers
    lr_w = 1e-5
    lr_b = 1e0
    lr_B = lr_b
    opt_w = tf.keras.optimizers.Adam(learning_rate=lr_w)
    opt_b = tf.keras.optimizers.Adam(learning_rate=lr_b)
    opt_B = tf.keras.optimizers.Adam(learning_rate=lr_B)
    # iterations parameters
    max_iters = 1000
    min_iters = int(0.05 * max_iters)
    min_grad_norm = 1e-3
    step = 0
    grad_norm = 1e9
    # regularizer parameter (for non-rigid)
    l = 0
    # number of cluster (masked non-rigid) or number of elements (fully non-rigid)
    num_clusters = tf.shape(x)[-1] if (mask is None) & (rigid == False) else None
    # Spatial transformer model
    model = SpatialTransformer1d(rigid=rigid, mask=mask, num_clusters=num_clusters, max_offset_range=max_offset_range)

    # fitting with a rough exploration (linear model) follow by gradient descent
    model = rough_exploration(model, x, target, init_params=[1., 0.], range_min=[-1e-2, -5000], range_max=[1e-2, 5000], debug_dir=debug_dir)
    while ((grad_norm > min_grad_norm) | (step < min_iters)) & (step < max_iters):
        loss_value, grads = train(model, x, target, l)
        grad_norm = tf.linalg.norm(tf.concat(grads, axis=-1))
        if debug_dir:
            if step % 10 == 0:
                if rigid:
                    print("\t\tnon-rigid - Step: {} - loss: {} - Params: W {}, b {}".format(step, loss_value.numpy(), model.W.numpy(), model.b.numpy()))
                    print("\t\t\tGrads: W {}, b {}, norm {}".format(np.mean(grads[0]), np.mean(grads[1]), grad_norm))
                else:
                    print("\t\tnon-rigid - Step: {} - loss: {} - Params: W {}, b {}, B {}".format(step
                                                                                                , loss_value.numpy()
                                                                                                , model.W.numpy()
                                                                                                , model.b.numpy()
                                                                                                , np.mean(model.B.numpy())))
                    print("\t\t\tGrads: W {}, b {}, B {}, norm {}".format(np.mean(grads[0]), np.mean(grads[1]), np.mean(grads[2]), grad_norm))
        opt_w.apply_gradients(zip(grads[:1], model.training_vars[:1]))
        opt_b.apply_gradients(zip(grads[1:2], model.training_vars[1:2]))
        if not rigid:
            opt_B.apply_gradients(zip(grads[2:], model.training_vars[2:]))
        step = step + 1

    # optimal parameters are the inverse of spatial transformer params
    if rigid:
        best_params = [1/model.training_vars[0].numpy(), (-1)*model.training_vars[1].numpy()]
    else:
        best_params = [1/model.training_vars[0].numpy(), (-1)*model.training_vars[1].numpy(), (-1)*model.training_vars[2].numpy()]
    if debug_dir:
        plot_data(model, x, target, debug_dir)

    return best_params