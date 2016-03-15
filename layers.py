import numpy as np
import theano
from theano import tensor

floatX = theano.config.floatX
np.random.seed(4444)


def tanh(x):
    return tensor.tanh(x)


def relu(x):
    tensor.nnet.relu(x)


def init_param(inp_size, out_size, name, scale=0.01, ortho=False):
    if ortho and inp_size == out_size:
        u, s, v = np.linalg.svd(np.random.randn(inp_size, inp_size))
        W = u.astype('float32')
    else:
        W = scale * np.random.randn(inp_size, out_size).astype(floatX)
    return theano.shared(W, name=name)


def init_bias(layer_size, name):
    return theano.shared(np.zeros(layer_size, dtype=floatX), name=name)


def _p(p, q, r):
    return '{}_{}_{}'.format(p, q, r)


class DropoutLayer(object):
    def __init__(self, p, trng):
        self.p = p
        self.trng = trng

    def fprop(self, x):
        retain_prob = 1. - self.p
        x *= self.trng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
        x /= retain_prob
        return x


class Sequence(object):
    def __init__(self, layers=None):
        self.layers = layers
        if layers is None:
            self.layers = list()

    def fprop(self, inp, **kwargs):
        z = inp
        for i, layer in enumerate(self.layers):
            z = layer(z, **kwargs)
        return z

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def add(self, layer):
        if isinstance(layer, list):
            self.layers += layer
        else:
            self.layers.append(layer)


class DenseLayer(object):
    def __init__(self, nin, dim, activ='lambda x: tensor.tanh(x)', prefix='ff',
                 postfix='0', scale=0.01, ortho=False, add_bias=True,
                 dropout=0., trng=None):
        self.activ = activ
        self.add_bias = add_bias
        self.W = init_param(nin, dim, _p(prefix, 'W', postfix),
                            scale=scale, ortho=ortho)
        if add_bias:
            self.b = init_bias(dim, _p(prefix, 'b', postfix))

        self.dropout_layer = None
        if dropout > 0.:
            self.dropout_layer = DropoutLayer(dropout, trng)

    def fprop(self, state_below, use_noise=False):
        pre_act = tensor.dot(state_below, self.W) + \
            (self.b if self.add_bias else 0.)
        z = eval(self.activ)(pre_act)
        if use_noise and self.dropout_layer is not None:
            z = self.dropout_layer.fprop(z)
        return z

    def get_params(self):
        params = {self.W.name: self.W}
        if self.add_bias:
            params[self.b.name] = self.b
        return params


class MultiLayer(object):
    def __init__(self, nin, dims, **kwargs):
        self.layers = []
        for i, dim in enumerate(dims):
            self.layers.append(DenseLayer(nin, dim, postfix=i, **kwargs))
            nin = dim

    def fprop(self, inp, **kwargs):
        for i, layer in enumerate(self.layers):
            inp = layer.fprop(inp, **kwargs)
        return inp

    def get_params(self):
        params = {}
        for layer in self.layers:
            params.update(**layer.get_params())
        return params


class Merger(object):
    def __init__(self, dims):
        self.dims = dims
        self.params = {}

    def fprop(self, inps, *args, **kwargs):
        return self._merge(inps, *args, **kwargs)

    def _merge(self, inps, axis=0, op='sum', **kwargs):
        if op == 'sum':
            merged = inps[0]
            for i in range(1, len(inps)):
                merged += inps[i]
            return merged
        else:
            raise ValueError("Unrecognized merge operation!")

    def get_params(self):
        return self.params
