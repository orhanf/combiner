import numpy as np
import theano
from theano import tensor

floatX = theano.config.floatX

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


class DenseLayer(object):
    def __init__(self, nin, dim, activ='lambda x: tensor.tanh(x)', prefix='ff',
                 postfix='0', scale=0.01, ortho=False, add_bias=True):
        self.activ = activ
        self.add_bias = add_bias
        self.W = init_param(nin, dim, _p(prefix, 'W', postfix),
                            scale=scale, ortho=ortho)
        if add_bias:
            self.b = init_bias(dim, _p(prefix, 'b', postfix))

    def fprop(self, state_below):
        pre_act = tensor.dot(state_below, self.W) + \
            (self.b if self.add_bias else 0.)
        return eval(self.activ)(pre_act)

    def get_params(self):
        return [self.W] + ([self.b] if self.add_bias else [])


class MultiLayer(object):
    def __init__(self, nin, dims, **kwargs):
        self.layers = []
        for i, dim in enumerate(dims):
            self.layers.append(DenseLayer(nin, dim, postfix=i, **kwargs))
            nin = dim

    def fprop(self, inp):
        for i, layer in enumerate(self.layers):
            inp = layer.fprop(inp)
        return inp

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


class Merger(object):
    def __init__(self, dims):
        self.dims = dims
        self.params = []

    def fprop(inps, *args, **kwargs):
        return self._merge(inps, *args, **kwargs)

    def _merge(self, inps, axis=0, op='sum'):
        if op == 'sum':
            merged = inps[0]
            for i in range(1, len(inps)):
                merged += inps[i]
            return merge
        else:
            raise ValueError("Unrecognized merge operation!")

    def get_params(self):
        return self.params


def build_model(xl, xr, y, learning_rate,
                input_dim_left, input_dim_right,
                left_layer_dims, right_layer_dims, top_layer_dims,
                merged_layer_dim):

    f_l = MultiLayer(input_dim_left, left_layer_dims)  # left branch
    f_r = MultiLayer(input_dim_right, right_layer_dims)  # right branch
    f_t = MultiLayer(merged_layer_dim, top_layer_dims)  # classifier
    merger = Merger(merged_layer_dim)  # merger layer

    fl = f_l.fprop(xl)
    fr = f_r.fprop(xr)

    fl_only = merger.fprop([fl])
    fr_only = merger.fprop([fr])
    f_both = merger.fprop([fl, fr], op='sum', axis=1)

    # left only
    yl_probs = tensor.nnet.softmax(f_t.fprop(fl_only))

    # right only
    yr_probs = tensor.nnet.softmax(f_t.fprop(fr_only))

    # both
    yb_probs = tensor.nnet.softmax(f_t.fprop(f_both))

    cost_yl = tensor.nnet.categorical_crossentropy(yl_probs, y).mean()
    cost_yr = tensor.nnet.categorical_crossentropy(yr_probs, y).mean()
    cost_yb = tensor.nnet.categorical_crossentropy(yb_probs, y).mean()

    cost_yl.name = 'cost_yl'
    cost_yr.name = 'cost_yr'
    cost_yb.name = 'cost_yb'

    params = f_l.get_params() + f_r.get_params() + f_t.get_params() + merger.get_params()

    grads = [theano.grad(cost, p) for p in params]
    updates = [(p, p - learning_rate * g) for p, g in zip(params, grads)]

    return cost_yl, cost_yr, cost_yb, params, grads, updates


def main():
    # spawn theano vars
    xl = tensor.matrix('xl')
    xr = tensor.matrix('xr')
    y = tensor.ivector('y')
    learning_rate = tensor.scalar('learning_rate')

    # use test values
    batch_size = 10
    theano.config.compute_test_value = 'raise'
    xl.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    xr.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    y.tag.test_value = np.random.randint(8, size=batch_size).astype(np.int32)
    learning_rate.tag.test_value = 0.5
    np.random.seed(4321)

    # build cgs
    cost_yl, cost_yr, cost_yb, param, grad, updates = build_model(
        xl, xr, y, learning_rate
        input_dim_left=2, input_dim_right=2,
        left_layer_dims=[3, 4], right_layer_dims=[7, 8], top_layer_dims=[7, 4],
        merged_layer_dim=5)

    # compile
    train = theano.function(inputs=[xl, xr, y, learning_rate],
                            outputs=[cost_yl, cost_yr, cost_yb],
                            updates=updates)

    # training loop
    niter = 1000
    learning_rate = np.float32(1.)
    for i in range(niter):
        xl_ = np.random.randn(batch_size, 2).astype(floatX)
        xr_ = np.random.randn(batch_size, 2).astype(floatX)
        y_ = np.random.randint(8, size=batch_size).astype(np.int32)

        cl, cr, cb = train(xl_, xr_, y_, learning_rate)
        print('iter: {} - cost: {} [label: {} domain: {}] - lambda_p: {}'
              .format(i, cl, cr, cb))

if __name__ == "__main__":
    main()
