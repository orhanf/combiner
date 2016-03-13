import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import MultiLayer, Merger
from optimizer import rmsprop
from training import train

floatX = theano.config.floatX


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
    yl_probs = T.nnet.softmax(f_t.fprop(fl_only))

    # right only
    yr_probs = T.nnet.softmax(f_t.fprop(fr_only))

    # both
    yb_probs = T.nnet.softmax(f_t.fprop(f_both))

    cost_l = T.nnet.categorical_crossentropy(yl_probs, y).mean()
    cost_r = T.nnet.categorical_crossentropy(yr_probs, y).mean()
    cost_b = T.nnet.categorical_crossentropy(yb_probs, y).mean()

    cost_l.name = 'cost_l'
    cost_r.name = 'cost_r'
    cost_b.name = 'cost_b'

    params_l = f_l.get_params() + f_t.get_params() + merger.get_params()
    params_r = f_r.get_params() + f_t.get_params() + merger.get_params()
    params_b = f_l.get_params() + f_r.get_params() + f_t.get_params() + \
        merger.get_params()

    grads_l = [theano.grad(cost_l, p) for p in params_l]
    grads_r = [theano.grad(cost_r, p) for p in params_r]
    grads_b = [theano.grad(cost_b, p) for p in params_b]

    acc_l = T.mean(T.eq(T.argmax(yl_probs, axis=1), y), dtype=floatX)
    acc_r = T.mean(T.eq(T.argmax(yr_probs, axis=1), y), dtype=floatX)
    acc_b = T.mean(T.eq(T.argmax(yb_probs, axis=1), y), dtype=floatX)

    model_l = (cost_l, acc_l, params_l, grads_l)
    model_r = (cost_r, acc_r, params_r, grads_r)
    model_b = (cost_b, acc_b, params_b, grads_b)

    return model_l, model_r, model_b


options = {
    'batch_size': 128,
    'nb_classes': 10,
    'lbranch': [],
    'rbranch': [],
    'tbranch': []
}


def main():
    # spawn theano vars
    xl = T.matrix('xl')
    xr = T.matrix('xr')
    y = T.ivector('y')
    learning_rate = T.scalar('learning_rate')
    trng = RandomStreams(1234)

    # use test values
    batch_size = 10
    theano.config.compute_test_value = 'raise'
    xl.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    xr.tag.test_value = np.random.randn(batch_size, 2).astype(floatX)
    y.tag.test_value = np.random.randint(8, size=batch_size).astype(np.int32)
    learning_rate.tag.test_value = 0.5
    np.random.seed(4321)

    # build cgs
    model_l, model_r, model_b = build_model(
        xl, xr, y, learning_rate,
        input_dim_left=2, input_dim_right=2,
        left_layer_dims=[3, 4], right_layer_dims=[7, 8], top_layer_dims=[7, 4],
        merged_layer_dim=5, trng=trng)

    # compile
    f_update_l = rmsprop(learning_rate, model_l, [xl, y])
    f_update_r = rmsprop(learning_rate, model_l, [xr, y])
    f_update_b = rmsprop(learning_rate, model_l, [xl, xr, y])

    # compile validation/test functions
    f_valid_l = theano.function([xl, y], [model_l[0], model_l[1]])
    f_valid_r = theano.function([xl, y], [model_r[0], model_r[1]])
    f_valid_b = theano.function([xl, y], [model_b[0], model_b[1]])

    # training loop
    train_err, valid_err, test_err = train(
        f_update_l, f_update_r, f_update_b,
        f_valid_l, f_valid_r, f_valid_b,
        xl, xr, y, learning_rate)

if __name__ == "__main__":
    main()
