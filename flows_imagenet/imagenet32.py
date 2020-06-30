import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal
from tqdm import tqdm
from logistic import mixlogistic_logpdf, mixlogistic_logcdf, mixlogistic_invcdf
from flow_training_imagenet import train, evaluate

DEFAULT_FLOATX = tf.float32
STORAGE_FLOATX = tf.float32

EPS = 1e-8

def to_default_floatx(x):
    return tf.cast(x, DEFAULT_FLOATX)

def at_least_float32(x):
    assert x.dtype in [tf.float16, tf.float32, tf.float64]
    if x.dtype == tf.float16:
        return tf.cast(x, tf.float32)
    return x

def get_var(var_name, *, ema, initializer, trainable=True, **kwargs):
    """forced storage dtype"""
    assert 'dtype' not in kwargs
    if isinstance(initializer, np.ndarray):
        initializer = initializer.astype(STORAGE_FLOATX.as_numpy_dtype)
    v = tf.get_variable(var_name, dtype=STORAGE_FLOATX, initializer=initializer, trainable=trainable, **kwargs)
    if ema is not None:
        assert isinstance(ema, tf.train.ExponentialMovingAverage)
        v = ema.average(v)
    return v

def _norm(x, *, axis, g, b, e=1e-5):
    assert x.shape.ndims == g.shape.ndims == b.shape.ndims
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.squared_difference(x, u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    return x * g + b

def norm(x, *, name, ema):
    """Layer norm over last axis"""
    with tf.variable_scope(name):
        dim = int(x.shape[-1])
        _g = get_var('g', ema=ema, shape=[dim], initializer=tf.constant_initializer(1))
        _b = get_var('b', ema=ema, shape=[dim], initializer=tf.constant_initializer(0))
        g, b = map(to_default_floatx, [_g, _b])
        bcast_shape = [1] * (x.shape.ndims - 1) + [dim]
        return _norm(x, g=tf.reshape(g, bcast_shape), b=tf.reshape(b, bcast_shape), axis=-1)

def int_shape(x):
    return list(map(int, x.shape.as_list())) 

def sumflat(x):
    return tf.reduce_sum(tf.reshape(x, [x.shape[0], -1]), axis=1)

def inverse_sigmoid(x):
    return -tf.log(tf.reciprocal(x) - 1.)

def init_normalization(x, *, name, init_scale=1., init, ema):
    with tf.variable_scope(name):
        g = get_var('g', shape=x.shape[1:], initializer=tf.constant_initializer(1.), ema=ema)
        b = get_var('b', shape=x.shape[1:], initializer=tf.constant_initializer(0.), ema=ema)
        if init:
            # data based normalization
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            assert m_init.shape == v_init.shape == scale_init.shape == g.shape == b.shape
            with tf.control_dependencies([
                g.assign(scale_init),
                b.assign(-m_init * scale_init)
            ]):
                g, b = tf.identity_n([g, b])
        return g, b

def dense(x, *, name, num_units, init_scale=1., init, ema):
    # use weight normalization (Salimans & Kingma, 2016)
    with tf.variable_scope(name):
        assert x.shape.ndims == 2
        _V = get_var('V', shape=[int(x.shape[1]), num_units], initializer=tf.random_normal_initializer(0, 0.05),
                     ema=ema)
        _g = get_var('g', shape=[num_units], initializer=tf.constant_initializer(1.), ema=ema)
        _b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)
        _vinvnorm = tf.rsqrt(tf.reduce_sum(tf.square(_V), [0]))
        V, g, b, vinvnorm = map(to_default_floatx, [_V, _g, _b, _vinvnorm])

        x0 = x = tf.matmul(x, V)
        x = (g * vinvnorm)[None, :] * x + b[None, :]

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            with tf.control_dependencies([
                _g.assign(tf.cast(g * scale_init, dtype=_g.dtype)),
                _b.assign_add(tf.cast(-m_init * scale_init, dtype=_b.dtype))
            ]):
                g, b = map(to_default_floatx, [_g, _b])
                x = (g * vinvnorm)[None, :] * x0 + b[None, :]

        return x

def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=(1, 1), pad='SAME', init_scale=1., init, ema):
    # use weight normalization (Salimans & Kingma, 2016)
    with tf.variable_scope(name):
        assert x.shape.ndims == 4
        _V = get_var('V', shape=[*filter_size, int(x.shape[-1]), num_units],
                     initializer=tf.random_normal_initializer(0, 0.05), ema=ema)
        _g = get_var('g', shape=[num_units], initializer=tf.constant_initializer(1.), ema=ema)
        _b = get_var('b', shape=[num_units], initializer=tf.constant_initializer(0.), ema=ema)
        _vnorm = tf.nn.l2_normalize(_V, [0, 1, 2])
        V, g, b, vnorm = map(to_default_floatx, [_V, _g, _b, _vnorm])

        W = g[None, None, None, :] * vnorm

        # calculate convolutional layer output
        input_x = x
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1, *stride, 1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale * tf.rsqrt(v_init + 1e-8)
            with tf.control_dependencies([
                _g.assign(tf.cast(g * scale_init, dtype=_g.dtype)),
                _b.assign_add(tf.cast(-m_init * scale_init, dtype=_b.dtype))
            ]):
                g, b = map(to_default_floatx, [_g, _b])
                W = g[None, None, None, :] * vnorm
                x = tf.nn.bias_add(tf.nn.conv2d(input_x, W, [1, *stride, 1], pad), b)

        return x

def nin(x, *, num_units, **kwargs):
    assert 'num_units' not in kwargs
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units=num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])

def matmul_last_axis(x, w):
    _, out_dim = w.shape
    s = x.shape.as_list()
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = tf.matmul(x, w)
    return tf.reshape(x, s[:-1] + [out_dim])

def concat_elu(x, *, axis=-1):
    return tf.nn.elu(tf.concat([x, -x], axis=axis))

def gate(x, *, axis):
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)

def gated_resnet(x, *, name, a, nonlinearity=concat_elu, conv=conv2d, use_nin, init, ema, dropout_p):
    with tf.variable_scope(name):
        num_filters = int(x.shape[-1])

        c1 = conv(nonlinearity(x), name='c1', num_units=num_filters, init=init, ema=ema)
        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), name='a_proj', num_units=num_filters, init=init, ema=ema)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)

        c2 = (nin if use_nin else conv)(c1, name='c2', num_units=num_filters * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)

def attn(x, *, name, pos_emb, heads, init, ema, dropout_p):
    with tf.variable_scope(name):
        bs, height, width, ch = x.shape.as_list()
        assert pos_emb.shape == [height, width, ch]
        assert ch % heads == 0
        timesteps = height * width
        dim = ch // heads
        # Position embeddings
        c = x + pos_emb[None, :, :, :]
        # b, h, t, d == batch, num heads, num timesteps, per-head dim (C // heads)
        c = nin(c, name='proj1', num_units=3 * ch, init=init, ema=ema)
        assert c.shape == [bs, height, width, 3 * ch]
        # Split into heads / Q / K / V
        c = tf.reshape(c, [bs, timesteps, 3, heads, dim])  # b, t, 3, h, d
        c = tf.transpose(c, [2, 0, 3, 1, 4])  # 3, b, h, t, d
        q_bhtd, k_bhtd, v_bhtd = tf.unstack(c, axis=0)
        assert q_bhtd.shape == k_bhtd.shape == v_bhtd.shape == [bs, heads, timesteps, dim]
        # Attention
        w_bhtt = tf.matmul(q_bhtd, k_bhtd, transpose_b=True) / np.sqrt(float(dim))
        w_bhtt = tf.cast(tf.nn.softmax(at_least_float32(w_bhtt)), dtype=x.dtype)
        assert w_bhtt.shape == [bs, heads, timesteps, timesteps]
        a_bhtd = tf.matmul(w_bhtt, v_bhtd)
        # Merge heads
        a_bthd = tf.transpose(a_bhtd, [0, 2, 1, 3])
        assert a_bthd.shape == [bs, timesteps, heads, dim]
        a_btc = tf.reshape(a_bthd, [bs, timesteps, ch])
        # Project
        c1 = tf.reshape(a_btc, [bs, height, width, ch])
        if dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = nin(c1, name='proj2', num_units=ch * 2, init_scale=0.1, init=init, ema=ema)
        return x + gate(c2, axis=3)

class Flow:
    def forward(self, x, **kwargs):
        raise NotImplementedError
    def backward(self, y, **kwargs):
        raise NotImplementedError

class Inverse(Flow):
    def __init__(self, base_flow):
        self.base_flow = base_flow

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x, **kwargs)
    
    def inverse(self, y, **kwargs):
        return self.base_flow.forward(y, **kwargs)

class Compose(Flow):
    def __init__(self, flows):
        self.flows = flows

    def _maybe_tqdm(self, iterable, desc, verbose):
        return tqdm(iterable, desc=desc) if verbose else iterable

    def forward(self, x, **kwargs):
        bs = int((x[0] if isinstance(x, tuple) else x).shape[0])
        logd_terms = []
        for i, f in enumerate(self._maybe_tqdm(self.flows, desc='forward {}'.format(kwargs),
                                               verbose=kwargs.get('verbose'))):
            assert isinstance(f, Flow)
            x, l = f.forward(x, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return x, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

    def inverse(self, y, **kwargs):
        bs = int((y[0] if isinstance(y, tuple) else y).shape[0])
        logd_terms = []
        for i, f in enumerate(
                self._maybe_tqdm(self.flows[::-1], desc='inverse {}'.format(kwargs), verbose=kwargs.get('verbose'))):
            assert isinstance(f, Flow)
            y, l = f.inverse(y, **kwargs)
            if l is not None:
                assert l.shape == [bs]
                logd_terms.append(l)
        return y, tf.add_n(logd_terms) if logd_terms else tf.constant(0.)

class ImgProc(Flow):
    def forward(self, x, **kwargs):
        x = x * (.9 / 256) + .05  # [0, 256] -> [.05, .95]
        x = -tf.log(1. / x - 1.)  # inverse sigmoid
        logd = np.log(.9 / 256) + tf.nn.softplus(x) + tf.nn.softplus(-x)
        logd = tf.reduce_sum(tf.reshape(logd, [int_shape(logd)[0], -1]), 1)
        return x, logd

    def inverse(self, y, **kwargs):
        y = tf.sigmoid(y)
        logd = tf.log(y) + tf.log(1. - y)
        y = (y - .05) / (.9 / 256)  # [.05, .95] -> [0, 256]
        logd -= np.log(.9 / 256)
        logd = tf.reduce_sum(tf.reshape(logd, [int_shape(logd)[0], -1]), 1)
        return y, logd

class TupleFlip(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tuple)
        a, b = x
        return (b, a), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return (b, a), None

class SpaceToDepth(Flow):
    def __init__(self, block_size=2):
        self.block_size = block_size

    def forward(self, x, **kwargs):
        return tf.space_to_depth(x, self.block_size), None

    def inverse(self, y, **kwargs):
        return tf.depth_to_space(y, self.block_size), None

class CheckerboardSplit(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tf.Tensor)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H, W // 2, 2, C])
        a, b = tf.unstack(x, axis=3)
        assert a.shape == b.shape == [B, H, W // 2, C]
        return (a, b), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        assert a.shape == b.shape
        B, H, W_half, C = a.shape
        x = tf.stack([a, b], axis=3)
        assert x.shape == [B, H, W_half, 2, C]
        return tf.reshape(x, [B, H, W_half * 2, C]), None

class ChannelSplit(Flow):
    def forward(self, x, **kwargs):
        assert isinstance(x, tf.Tensor)
        assert len(x.shape) == 4 and x.shape[3] % 2 == 0
        return tuple(tf.split(x, 2, axis=3)), None

    def inverse(self, y, **kwargs):
        assert isinstance(y, tuple)
        a, b = y
        return tf.concat([a, b], axis=3), None

class Sigmoid(Flow):
    def forward(self, x, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        return y, sumflat(logd)
    def inverse(self, y, **kwargs):
        x = inverse_sigmoid(y)
        logd = -tf.log(y) - tf.log(1. - y)
        return x, sumflat(logd)

class SmoothSigmoid(Flow):
    '''
    This class is for fixing the gradient NAN bug in Flow++.
    See Appendix.B in our paper https://arxiv.org/abs/2002.09741
    '''
    def forward(self, x, **kwargs):
        y = tf.sigmoid(x)
        logd = -tf.nn.softplus(x) - tf.nn.softplus(-x)
        y = (y - 0.05) / 0.9
        logd += -tf.log(0.9)
        return y, sumflat(logd)

    def inverse(self, y, **kwargs):
        y = y * 0.9 + 0.05
        x = inverse_sigmoid(y)
        logd = -tf.log(y + EPS) - tf.log(1. - y + EPS)
        logd += tf.log(0.9)
        return x, sumflat(logd)

class Norm(Flow):
    def __init__(self, init_scale=1.):
        def f(input_, forward, init, ema):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            bs = int(input_[0].shape[0])
            g_and_b = []
            for (i, x) in enumerate(input_):
                g, b = init_normalization(x, name='norm{}'.format(i), init_scale=init_scale, init=init, ema=ema)
                g = tf.maximum(g, 1e-10)
                assert x.shape[0] == bs and g.shape == b.shape == x.shape[1:]
                g_and_b.append((g, b))

            logd = tf.fill([bs], tf.add_n([tf.reduce_sum(tf.log(g)) for (g, _) in g_and_b]))
            if forward:
                out = [x * g[None] + b[None] for (x, (g, b)) in zip(input_, g_and_b)]
            else:
                out = [(x - b[None]) / g[None] for (x, (g, b)) in zip(input_, g_and_b)]
                logd = -logd

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, **kwargs):
        return self.template(x, forward=True, init=init, ema=ema)

    def inverse(self, y, init=False, ema=None, **kwargs):
        return self.template(y, forward=False, init=init, ema=ema)            

class Pointwise(Flow):
    def __init__(self, noisy_identity_init=0.01):
        def f(input_, forward, init, ema):
            assert not isinstance(input_, list)
            if isinstance(input_, tuple):
                is_tuple = True
            else:
                assert isinstance(input_, tf.Tensor)
                input_ = [input_]
                is_tuple = False

            out, logds = [], []
            for i, x in enumerate(input_):
                _, img_h, img_w, img_c = x.shape.as_list()
                if noisy_identity_init:
                    # identity + gaussian noise
                    initializer = (
                            np.eye(img_c) + noisy_identity_init * np.random.randn(img_c, img_c)
                    ).astype(np.float32)
                else:
                    init_w = np.random.randn(img_c, img_c)
                    initializer = np.linalg.qr(init_w)[0].astype(np.float32)
                W = get_var('W{}'.format(i), ema=ema, shape=None, initializer=initializer)
                out.append(self._nin(x, W if forward else tf.matrix_inverse(W)))
                logds.append(
                    (1 if forward else -1) * img_h * img_w *
                    tf.cast(tf.log(tf.abs(tf.matrix_determinant(tf.cast(W, tf.float64))) + 1e-8), tf.float32)
                )
            logd = tf.fill([input_[0].shape[0]], tf.add_n(logds))

            if not is_tuple:
                assert len(out) == 1
                return out[0], logd
            return tuple(out), logd

        self.template = tf.make_template(self.__class__.__name__, f)

    @staticmethod
    def _nin(x, w, b=None):
        _, out_dim = w.shape
        s = x.shape.as_list()
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = tf.matmul(x, w)
        if b is not None:
            assert b.shape.ndims == 1
            x = x + b[None, :]
        return tf.reshape(x, s[:-1] + [out_dim])

    def forward(self, x, init=False, ema=None, **kwargs):
        return self.template(x, forward=True, init=init, ema=ema)

    def inverse(self, y, init=False, ema=None, **kwargs):
        return self.template(y, forward=False, init=init, ema=ema)

class MixLogisticCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, use_nin, components, attn_heads, use_ln,
                 with_affine=True, use_final_nin=False, init_scale=0.1, nonlinearity=concat_elu):
        self.components = components
        self.with_affine = with_affine
        self.scale_flow = Inverse(SmoothSigmoid())

        def f(x, init, ema, dropout_p, verbose, context):
            # if verbose and context is not None:
            #     print('got context')
            if init and verbose:
                # debug stuff
                with tf.variable_scope('debug'):
                    xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
                    x = tf.Print(
                        x,
                        [
                            tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x),
                            tf.reduce_any(tf.is_nan(x)), tf.reduce_any(tf.is_inf(x))
                        ],
                        message='{} (shape/mean/std/min/max/nan/inf) '.format(self.template.variable_scope.name),
                        summarize=10,
                    )
            B, H, W, C = x.shape.as_list()

            pos_emb = to_default_floatx(get_var(
                'pos_emb', ema=ema, shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
            ))
            x = conv2d(x, name='c1', num_units=filters, init=init, ema=ema)
            for i_block in range(blocks):
                with tf.variable_scope('block{}'.format(i_block)):
                    x = gated_resnet(
                        x, name='conv', a=context, use_nin=use_nin, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln1', ema=ema)
            x = nonlinearity(x)
            x = (nin if use_final_nin else conv2d)(
                x, name='c2', num_units=C * (2 + 3 * components), init_scale=init_scale, init=init, ema=ema
            )
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            x = at_least_float32(x)  # do mix-logistics in tf.float32

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            ml_logscales = tf.maximum(ml_logscales, -7.)

            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]
            return s, t, ml_logits, ml_means, ml_logscales

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = tf.exp(
            mixlogistic_logcdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        )
        out, scale_logd = self.scale_flow.forward(out)
        if self.with_affine:
            assert out.shape == s.shape == t.shape
            out = tf.exp(s) * out + t

        logd = mixlogistic_logpdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert scale_logd.shape == logd.shape
        logd += scale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

    def inverse(self, y, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = float_ef
        if self.with_affine:
            out = tf.exp(-s) * (out - t)
        out, invscale_logd = self.scale_flow.inverse(out)
        out = tf.clip_by_value(out, 1e-5, 1. - 1e-5)
        out = mixlogistic_invcdf(y=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)

        logd = mixlogistic_logpdf(x=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = -tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert invscale_logd.shape == logd.shape
        logd += invscale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

class MixLogisticAttnCoupling(Flow):
    """
    CDF of mixture of logistics, followed by affine
    """

    def __init__(self, filters, blocks, use_nin, components, attn_heads, use_ln,
                 with_affine=True, use_final_nin=False, init_scale=0.1, nonlinearity=concat_elu):
        self.components = components
        self.with_affine = with_affine
        self.scale_flow = Inverse(SmoothSigmoid())

        def f(x, init, ema, dropout_p, verbose, context):
            if init and verbose:
                with tf.variable_scope('debug'):
                    xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
                    x = tf.Print(
                        x,
                        [
                            tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x),
                            tf.reduce_any(tf.is_nan(x)), tf.reduce_any(tf.is_inf(x))
                        ],
                        message='{} (shape/mean/std/min/max/nan/inf) '.format(self.template.variable_scope.name),
                        summarize=10,
                    )
            B, H, W, C = x.shape.as_list()

            pos_emb = to_default_floatx(get_var(
                'pos_emb', ema=ema, shape=[H, W, filters], initializer=tf.random_normal_initializer(stddev=0.01),
            ))
            x = conv2d(x, name='c1', num_units=filters, init=init, ema=ema)
            for i_block in range(blocks):
                with tf.variable_scope('block{}'.format(i_block)):
                    x = gated_resnet(
                        x, name='conv', a=context, use_nin=use_nin, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln1', ema=ema)
                    x = attn(
                        x, name='attn', pos_emb=pos_emb, heads=attn_heads, init=init, ema=ema, dropout_p=dropout_p
                    )
                    if use_ln:
                        x = norm(x, name='ln2', ema=ema)
                    assert x.shape == [B, H, W, filters]
            x = nonlinearity(x)
            x = (nin if use_final_nin else conv2d)(
                x, name='c2', num_units=C * (2 + 3 * components), init_scale=init_scale, init=init, ema=ema
            )
            assert x.shape == [B, H, W, C * (2 + 3 * components)]
            x = tf.reshape(x, [B, H, W, C, 2 + 3 * components])

            x = at_least_float32(x)  # do mix-logistics stuff in float32

            s, t = tf.tanh(x[:, :, :, :, 0]), x[:, :, :, :, 1]
            ml_logits, ml_means, ml_logscales = tf.split(x[:, :, :, :, 2:], 3, axis=4)
            ml_logscales = tf.maximum(ml_logscales, -7.)

            assert s.shape == t.shape == [B, H, W, C]
            assert ml_logits.shape == ml_means.shape == ml_logscales.shape == [B, H, W, C, components]
            return s, t, ml_logits, ml_means, ml_logscales

        self.template = tf.make_template(self.__class__.__name__, f)

    def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(x, tuple)
        cf, ef = x
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        # my_cast = lambda m: tf.cast(m, tf.float64) if len(m.shape) > 0 else m
        # s,t,ml_logits, ml_means, ml_logscales, float_ef = map(my_cast, [s,t,ml_logits, ml_means, ml_logscales, float_ef])

        out = tf.exp(
            mixlogistic_logcdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        )
        out, scale_logd = self.scale_flow.forward(out)
        if self.with_affine:
            assert out.shape == s.shape == t.shape
            out = tf.exp(s) * out + t

        logd = mixlogistic_logpdf(x=float_ef, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert scale_logd.shape == logd.shape
        logd += scale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

    def inverse(self, y, init=False, ema=None, dropout_p=0., verbose=True, context=None, **kwargs):
        assert isinstance(y, tuple)
        cf, ef = y
        float_ef = at_least_float32(ef)
        s, t, ml_logits, ml_means, ml_logscales = self.template(
            cf, init=init, ema=ema, dropout_p=dropout_p, verbose=verbose, context=context
        )

        out = float_ef
        if self.with_affine:
            out = tf.exp(-s) * (out - t)
        out, invscale_logd = self.scale_flow.inverse(out)
        out = tf.clip_by_value(out, 1e-5, 1. - 1e-5)
        out = mixlogistic_invcdf(y=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)

        logd = mixlogistic_logpdf(x=out, prior_logits=ml_logits, means=ml_means, logscales=ml_logscales)
        if self.with_affine:
            assert s.shape == logd.shape
            logd += s
        logd = -tf.reduce_sum(tf.layers.flatten(logd), axis=1)
        assert invscale_logd.shape == logd.shape
        logd += invscale_logd

        out, logd = map(to_default_floatx, [out, logd])
        assert out.shape == ef.shape == cf.shape and out.dtype == ef.dtype == logd.dtype == cf.dtype
        return (cf, out), logd

def gaussian_sample_logp(shape, dtype):
    eps = tf.random_normal(shape)
    logp = Normal(0., 1.).log_prob(eps)
    assert logp.shape == eps.shape
    logp = tf.reduce_sum(tf.layers.flatten(logp), axis=1)
    return tf.cast(eps, dtype=dtype), tf.cast(logp, dtype=dtype)

def construct(*, filters, blocks, components, attn_heads, use_nin, use_ln, extra_dims, x_dims=3):
    dequant_coupling_kwargs = dict(
        filters=filters, blocks=8, use_nin=use_nin, components=components, attn_heads=attn_heads, use_ln=use_ln
    )
    posterior_coupling_kwargs = dict(
        filters=filters, blocks=8, use_nin=use_nin, components=components, attn_heads=attn_heads, use_ln=use_ln
    )
    coupling_kwargs = dict(
        filters=filters, blocks=blocks, use_nin=use_nin, components=components, attn_heads=attn_heads, use_ln=use_ln
    )

    class NormalPosterior(Flow):
        def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, **kwargs):
            z, logq_z = gaussian_sample_logp(x.shape.as_list()[:-1] + [extra_dims], dtype=DEFAULT_FLOATX)
            return z, tf.reshape(logq_z, [-1])


    class FlowPosterior(Flow):
        def __init__(self, q_flow):
            super().__init__()
            self.q_flow = q_flow

            def deep_processor(x, *, init, ema, dropout_p):
                (this, that), _ = CheckerboardSplit().forward(x)
                processed_context = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, init=init, ema=ema)
                B, H, W, C = processed_context.shape.as_list()

                pos_emb = to_default_floatx(get_var(
                    'pos_emb_dq', ema=ema, shape=[H, W, C], initializer=tf.random_normal_initializer(stddev=0.01),
                ))

                for i in range(8):
                    processed_context = gated_resnet(
                        processed_context, name=f'c{i}',
                        a=None, dropout_p=dropout_p, ema=ema, init=init,
                        use_nin=False
                    )
                    processed_context = norm(processed_context, name=f'dqln{i}', ema=ema)
                    processed_context = attn(processed_context, name=f'dqattn{i}', pos_emb=pos_emb, heads=4, init=init, ema=ema, dropout_p=dropout_p)
                    processed_context = norm(processed_context, name=f'ln{i}', ema=ema)
                    
                return processed_context

            self.context_proc = tf.make_template("q_context_proc", deep_processor)

        def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, **kwargs):
            '''
            z = f(u)
            u = f^{-1}(z)
            p(u) = p(z) * |det(J_f^{-1}^{-1}(u))| = p(z) * |det(J_f(u))|
            '''
            eps, eps_logp = gaussian_sample_logp(x.shape.as_list()[:-1] + [extra_dims], dtype=DEFAULT_FLOATX)
            xd, logd = self.q_flow.forward(
                eps,
                context=self.context_proc(x, init=init, ema=ema, dropout_p=dropout_p),
                init=init, ema=ema, dropout_p=dropout_p, verbose=verbose
            )
            z, logq_z = xd, eps_logp - logd
            return z, tf.reshape(logq_z, [-1])


    class Dequantizer(Flow):
        def __init__(self, dequant_flow):
            super().__init__()
            assert isinstance(dequant_flow, Flow)
            self.dequant_flow = dequant_flow

            def deep_processor(x, *, init, ema, dropout_p):
                (this, that), _ = CheckerboardSplit().forward(x)
                processed_context = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, init=init, ema=ema)
                B, H, W, C = processed_context.shape.as_list()

                pos_emb = to_default_floatx(get_var(
                    'pos_emb_dq', ema=ema, shape=[H, W, C], initializer=tf.random_normal_initializer(stddev=0.01),
                ))

                for i in range(8):
                    processed_context = gated_resnet(
                        processed_context, name=f'c{i}',
                        a=None, dropout_p=dropout_p, ema=ema, init=init,
                        use_nin=False
                    )
                    processed_context = norm(processed_context, name=f'dqln{i}', ema=ema)
                    processed_context = attn(processed_context, name=f'dqattn{i}', pos_emb=pos_emb, heads=4, init=init, ema=ema, dropout_p=dropout_p)
                    processed_context = norm(processed_context, name=f'ln{i}', ema=ema)
                    
                return processed_context

            self.context_proc = tf.make_template("context_proc", deep_processor)

        def forward(self, x, init=False, ema=None, dropout_p=0., verbose=True, **kwargs):
            eps, eps_logli = gaussian_sample_logp(x.shape, dtype=DEFAULT_FLOATX)
            xd, logd = self.dequant_flow.forward(
                eps,
                context=self.context_proc(x / 256.0 - 0.5, init=init, ema=ema, dropout_p=dropout_p),
                init=init, ema=ema, dropout_p=dropout_p, verbose=verbose
            )
            assert x.shape == xd.shape and logd.shape == eps_logli.shape
            x, dequant_logd = x + xd, logd - eps_logli
            x, logd = ImgProc().forward(x)
            return x, dequant_logd + logd


    class Generative(Flow):
        def __init__(self, flow):
            super().__init__()
            self.flow = flow

        def forward(self, x, z, flow_kwargs):
            '''
            y = f(xz)
            xz = f^{-1}(y)
            p(xz) = p(y) * |det(J_f^{-1}^{-1}(xz))| = p(y) * |det(J_f(xz))|
            '''
            H, W, Cx = x.shape.as_list()[-3:]
            x = tf.reshape(x, [-1, H, W, Cx])
            z = tf.reshape(z, [-1, H, W, extra_dims])

            xz = tf.concat([x, z], axis=-1)
            y, logd_flow = self.flow.forward(xz, **flow_kwargs)
            logpy = sumflat(Normal(0., 1.).log_prob(y))
            return y, logpy + logd_flow

        def inverse(self, y, flow_kwargs):
            '''
            g = f^{-1}
            y = g^{-1}(xz)
            p(y) = p(xz) * |det(J_g^{-1}^{-1}(y))| = p(xz) * |det(J_g(y))|
            '''
            y = tf.reshape(y, [-1] + y.shape.as_list()[-3:])
            logpy = sumflat(Normal(0., 1.).log_prob(y))

            xz = y
            xz = tf.reshape(xz, [-1, 16, 16, 4*(x_dims+extra_dims)])
            xz, logd_flow = self.flow.inverse(xz, **flow_kwargs)
            return xz, logpy + logd_flow

        def sample(self, num_samples, flow_kwargs):
            y = tf.random_normal([num_samples, 32, 32, x_dims + extra_dims])
            xz, logp_xz = self.inverse(y, flow_kwargs)
            x = xz[:,:,:,:x_dims]
            x, logd = ImgProc().inverse(x)
            return x, logp_xz + logd

    dequant_flow = Dequantizer(Compose([
        CheckerboardSplit(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        CheckerboardSplit(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        Sigmoid(),
    ]))

    posterior_flow = FlowPosterior(Compose([
        CheckerboardSplit(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        CheckerboardSplit(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Norm(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        Sigmoid(),
    ]))

    flow = Generative(Compose([
        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),

        SpaceToDepth(),

        ChannelSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(ChannelSplit()),

        CheckerboardSplit(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
        Inverse(CheckerboardSplit()),
    ]))

    return dequant_flow, flow, posterior_flow


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    args = parser.parse_args()

    global DEFAULT_FLOATX
    DEFAULT_FLOATX = tf.float32

    max_lr = 1e-3
    min_lr = 3e-4
    warmup_steps = 5000
    bs = int(32)

    def lr_schedule(step, *, decay=0.99995):
        global curr_lr
        try:
            if step < warmup_steps:
                curr_lr = max_lr * step / warmup_steps
                return max_lr * step / warmup_steps
            elif step > (warmup_steps * 10) and curr_lr >  min_lr:
                curr_lr *= decay
                return curr_lr
            return curr_lr
        except Exception:
            curr_lr = min_lr
            return min_lr

    dropout_p = 0.
    filters = 128
    blocks = 20
    components = 32  # logistic mixture components
    attn_heads = 4
    use_ln = True

    extra_dims = 3

    floatx_str = {tf.float32: 'fp32', tf.float16: 'fp16'}[DEFAULT_FLOATX]

    if args.eval_checkpoint:
        evaluate(flow_constructor=lambda: construct(
            filters=filters,
            components=components,
            attn_heads=attn_heads,
            blocks=blocks,
            use_nin=True,
            use_ln=use_ln,
            extra_dims=extra_dims,
        ), seed=0, restore_checkpoint=args.eval_checkpoint, iw_samples=1024, total_bs=32)
        return

    train(
        flow_constructor=lambda: construct(
            filters=filters,
            components=components,
            attn_heads=attn_heads,
            blocks=blocks,
            use_nin=True,
            use_ln=use_ln,
            extra_dims=extra_dims,
        ),
        logdir=f'~/logs/vflow_imagenet32_mix{components}_b{blocks}_f{filters}_h{attn_heads}_ln{int(use_ln)}_lr{max_lr}_bs{bs}_drop{dropout_p}_{floatx_str}',
        lr_schedule=lr_schedule,
        dropout_p=dropout_p,
        seed=0,
        init_bs=bs,
        dataset='imagenet32',
        total_bs=bs,
        ema_decay=.999222,
        steps_per_log=100,
        steps_per_dump=10000,
        steps_per_samples=10000,
        max_grad_norm=1.,
        dtype=DEFAULT_FLOATX,
        scale_loss=1e-2 if DEFAULT_FLOATX == tf.float16 else None,
        n_epochs=2,
        restore_checkpoint=None, # put in path to checkpoint in the format: path_to_checkpoint/model (no .meta / .ckpt)
        dump_samples_to_tensorboard=True, # if you want to push the tiled simples to tensorboard. 
    )


if __name__ == '__main__':
    main()
