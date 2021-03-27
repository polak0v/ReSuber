import tensorflow as tf

class SpatialTransformer1d():
    """ Derivable 1D rigid/non-rigid transformation of the input

    A 1D spatial transformer is used to make a rigid/non-rigid transformation of the source 1D vector 
    onto the target 1D vector given the transformation parameters. The non-rigid transformation can be
    masked (cluster-wise) or not (input element-wise).

    Attributes
    ----------
        rigid : bool
            describes if the tranformation is rigid or not (default: True)
        eps : float
            a small value instead of zero is used to avoid numerical issue in the gradient computation (default: 1e-9)
        W : `list` [`float`]
            contains the polynomial trainable weights, 1st element is the highest order (default: 1. + eps)
        b : float
            trainable offset of the polynomial (default: 0. + eps)
        B : `list` [`float`]
            non-rigid displacement parameters
        training_vars : `list` [`tf.Variable`]
            the training variables of the model
        mask: `list` [`int`]
            used in masked non-rigid transformation to specify the different clusters on the input
        num_clusters : int
            number of clusters that will be transformed, should be the input size for non-masked non-rigid transformation
        max_offset_range : float
            maximum offset range allowed for non-rigid transformation (default: None)
    """

    def __init__(self, rigid=True, mask=None, num_clusters=None, max_offset_range=None, w_trainable=True, b_trainable=True):
        """Initialize the SpatialTransformer1d class.

        Parameters
        ----------
            rigid : bool
                describes if the tranformation is rigid or not (default: True)
            mask: vector of ints
                used in masked non-rigid transformation to specify the different clusters on the input
            num_clusters : int
                number of clusters that will be transformed, requires to be the input size for non-masked non-rigid transformation
            max_offset_range : float
                maximum offset range allowed for non-rigid transformation (default: None)
            w_trainable : bool
                trainable weight or not (default: True)
            b_trainable : bool
                trainable bias or not (default: True)
        """
        self.rigid = rigid
        self.eps = 1e-9
        self.W = tf.Variable([1. + self.eps], dtype=tf.float32, trainable=w_trainable, name="W")
        self.b = tf.Variable([0. + self.eps], dtype=tf.float32, trainable=b_trainable, name="b")
        self.max_offset_range = max_offset_range
        self.num_clusters = num_clusters
        self.mask = mask
        if not self.rigid:
            if self.mask is not None:
                self.num_clusters = tf.cast(tf.reduce_max(self.mask) + 1, dtype=tf.int32)
            if max_offset_range is not None:
                self.B = tf.Variable(self.eps * tf.random.uniform(shape=[self.num_clusters])
                                    , constraint=lambda x: tf.clip_by_value(x, (-1.)*self.max_offset_range, self.max_offset_range)
                                    , dtype=tf.float32, trainable=True, name="B")
            else:
                self.B = tf.Variable(self.eps * tf.random.uniform(shape=[self.num_clusters]), dtype=tf.float32, trainable=True, name="B")                
            self.training_vars = [self.W, self.b, self.B]
        else:
            self.training_vars = [self.W, self.b]
    
    def update_params(self, W=None, b=None, B=None):
        """Update the learning parameters of the model.

        Parameters
        ----------
            W : vector of floats
                contains the polynomial trainable weights, 1st element is the highest order
            b : float
                trainable offset of the polynomial
            B : vector of floats
                non-rigid displacement parameters
        """
        if W is not None:
            self.W.assign(W)
        if b is not None:
            self.b.assign(b)
        if B is not None:
            self.B.assign(B)
    
    @tf.function
    def resample(self, inp):
        """Resample the input onto the target positions.

        First, the target positions are calculated from a grid that was linearly transformed using the polynomials parameters.
        It will then resample the input onto the target positions using linear interpolation.
        If configured, it will be followed by a non-linear registration (cluster or element-wise).

        Parameters
        ----------
            inp : tf.tensor of tf.float32 with shape [N]
                The input source 1D vector to be transformed.

        Returns
        -------
            tf.tensor of tf.float32 with shape [N] : resampled input onto target positions
        """
        self.input_size = tf.shape(inp)[-1]
        min_max_ref_grid = tf.stack([0., tf.cast(self.input_size, tf.float32) - 1.], axis=0)

        transformed_grid = self.transform_grid_rigid(min_max_ref_grid=min_max_ref_grid)
        if self.rigid:
            output = self.interpolate(inp=inp, points=transformed_grid, min_max_ref_grid=min_max_ref_grid)
        else:
            non_rigid_grid = self.transform_grid_non_rigid(rigid_transformed_grid=transformed_grid, min_max_ref_grid=min_max_ref_grid)
            output = self.interpolate(inp=inp, points=non_rigid_grid, min_max_ref_grid=min_max_ref_grid)
            
        return output

    def transform_grid_rigid(self, min_max_ref_grid):
        """Rigid transform of the source reference grid.

        Parameters
        ----------
            min_max_ref_grid : tf.tensor of tf.float32 with shape [2]
                min and max values of the reference grid

        Returns
        -------
            tf.tensor of tf.float32 with shape [N] : transformed grid
        """
        mx = tf.linspace(min_max_ref_grid[0], min_max_ref_grid[-1], self.input_size)
        output = tf.math.polyval([*tf.unstack(self.W), *tf.unstack(self.b)], mx)
        return output

    def transform_grid_non_rigid(self, rigid_transformed_grid, min_max_ref_grid):
        """Non-rigid transform of the rigid transformed source reference grid.

        Parameters
        ----------
            rigid_transformed_grid : tf.tensor of tf.float32 with shape [N]
                rigid transformed source reference grid
            min_max_ref_grid : tf.tensor of tf.float32 with shape [2]
                min and max values of the reference grid

        Returns
        -------
            tf.tensor of tf.float32 with shape [N] : transformed grid
        """
        output = tf.zeros(self.input_size, dtype=tf.float32)
        if self.mask is not None:
            # resample the mask
            resampled_mask = self.interpolate(tf.cast(self.mask, dtype=tf.float32), rigid_transformed_grid, min_max_ref_grid)
            # for each cluster, apply the corresponding displacement
            for c in range(self.num_clusters):
                mx_mask = tf.equal(tf.cast(resampled_mask, dtype=tf.int32), c)
                if tf.reduce_any(mx_mask):
                    # min_max = tf.cast(tf.where(mx_mask), dtype=tf.float32)
                    # transformed_mask = tf.clip_by_value(rigid_transformed_grid + self.B[c], min_max[0], min_max[-1])
                    # output = output + transformed_mask * tf.cast(mx_mask, dtype=tf.float32)
                    output = output + (rigid_transformed_grid + self.B[c])* tf.cast(mx_mask, dtype=tf.float32)
        else:
            # if the mask is not defined, each element has his own displacement
            output = rigid_transformed_grid + self.B
        return output
    
    def interpolate(self, inp, points, min_max_ref_grid):
        """Linear interpolation of the input onto the positions.

        Applies a linear interpolation using the input values and the target positions.
        It uses a zero-padding mode, so all points outside of the bounded values will be zero.

        Parameters
        ----------
            inp : `tf.tensor` [`tf.float32`]
                input source 1D vector to be interpolated.
            points : `tf.tensor` [`tf.float32`]
                target positions to interpolate on
            min_max_ref_grid : `tf.tensor` [`tf.float32`]
                bounding value for the interpolation

        Returns
        -------
            `tf.tensor` [`tf.float32`] : interpolated values at points
        """
        # nearest neighboor interpolation method
        width = tf.cast(self.input_size, dtype=tf.float32)
        width_i = tf.cast(width, dtype=tf.int32)
        zero = tf.zeros([], dtype=tf.float32)
        zero_i = tf.cast(zero, dtype=tf.int32)
        output = tf.zeros(self.input_size, dtype=tf.float32)
        min_max_ref_grid = min_max_ref_grid

        # scale positions to [0, width/height - 1]
        coeff_x = (width - 1.)/(min_max_ref_grid[-1] - min_max_ref_grid[0])
        ix = (coeff_x * points) - (coeff_x * min_max_ref_grid[0])

        # zeros padding mode, for positions outside of refrence grid
        valid = tf.cast(tf.less_equal(ix, width - 1.) & tf.greater_equal(ix, zero), dtype=tf.float32)

        # get lef corner indexes based on the scaled positions
        ix_left = tf.clip_by_value(tf.floor(ix), zero, width - 1.)
        ix_right = tf.clip_by_value(tf.floor(ix) + 1., zero, width - 1.)
        ix_left_i = tf.cast(ix_left, dtype=tf.int32)
        ix_right_i = tf.cast(ix_right, dtype=tf.int32)

        # rounded (derivable) distance to each opposite side
        wr = tf.abs((ix - ix_left))
        wl = 1 - wr

        # gather input image values from corners idx, and calculate weighted pixel value
        idx_left = tf.clip_by_value(ix_left_i, zero_i, width_i - 1)
        idx_right = tf.clip_by_value(ix_right_i, zero_i, width_i - 1)
        Il = tf.gather(inp, idx_left)
        Ir = tf.gather(inp, idx_right)

        output = tf.math.accumulate_n([wl * Il, wr * Ir]) * valid

        return output
    
    def __call__(self, inputs):
        output = self.resample(inputs)
        return output