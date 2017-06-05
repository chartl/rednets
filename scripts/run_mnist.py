#### my own mnist example

from __future__ import print_function

from argparse import ArgumentParser

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as tt

import gzip

import lasagne


def get_args():
    parser = ArgumentParser('run_mnist')
    parser.add_argument('epoch_log', help='The file for the accuracy output by epoch', type=str)
    parser.add_argument('--conv', help='Comma-separated size:stride:filter:pad of convolutional layers', type=str, 
        default='5:2:5:1,5:2:50:0')
    parser.add_argument('--deep', help='Comma-separated size of deep layers (not including 10-node output', type=str, default='100')
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)

    args = parser.parse_args()

    convolutions = [dict(zip(['filter_size', 'stride', 'num_filters', 'pad'], map(int, x.split(':'))))
                    for x in args.conv.split(',')]


    # check that the convolutions are even possible
    in_square, min_square = 28, 3  # input is 28 x 28; minimum output of the conv net is 3 x 3

    print('Architecture::Input 28 x 28 x 1')
    print('Architecture::Convolutions')
    for i, conv_params in enumerate(convolutions):
        out_square = 1 + (in_square - conv_params['filter_size'] + 2 * conv_params['pad'])/conv_params['stride']
        if out_square < min_square:
            raise ValueError('Error in convolutional parameters: layer size became < {}'.format(min_square))
        print('Layer {}: {} x {} x {}'.format(i + 1, out_square, out_square, conv_params['num_filters']))
        in_square = out_square

    full = [{'num_units': int(x)} for x in args.deep.split(',')]

    print('Architecture::Dense')
    for i, dparams in enumerate(full):
        print('Layer {}: {}'.format(i + len(convolutions) + 1, dparams['num_units']))

    return convolutions, full, args.epochs, args.epoch_log


def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def add_conv_layer(prev_layer, dropout=None, **conv2d_kwargs):
    """\
    Given a previous layer, hook up a 2d convolutional layer, with arguments
    given in `conv2d_kwargs`.

    :dropout: If a fraction, make this a dropout layer with the given p(dropout)

    """
    if 'W' not in conv2d_kwargs:
        conv2d_kwargs['W'] = lasagne.init.GlorotUniform()
    if 'nonlinearity' not in conv2d_kwargs:
        conv2d_kwargs['nonlinearity'] = lasagne.nonlinearities.rectify

    next_layer = lasagne.layers.Conv2DLayer(prev_layer, **conv2d_kwargs)
    if dropout is not None and conv2d_kwargs['filter_size'] > 1:
        next_layer = lasagne.layers.DropoutLayer(next_layer, p=dropout)

    return next_layer


def add_full_layer(prev_layer, dropout=None, **full_kwargs):
    """\
    Given a previous layer, hook up a fully-connected layer
    with the arguments given in `full_kwargs`

    :dropout: If a float, make this a droupout layer with the given p(dropout)

    """
    if 'W' not in full_kwargs:
        full_kwargs['W'] = lasagne.init.GlorotUniform()
    if 'nonlinearity' not in full_kwargs:
        full_kwargs['nonlinearity'] = lasagne.nonlinearities.sigmoid

    next_layer = lasagne.layers.DenseLayer(prev_layer, **full_kwargs)

    if dropout is not None:
        next_layer = lasagne.layers.DropoutLayer(next_layer, p=dropout)

    return next_layer

def main():
    conv_params, full_params, epochs, logfile = get_args()

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    ## first, try out a simple conv-pool-full network
    image_var = tt.tensor4('mnist_input')
    label_var = tt.ivector('mnist_labels')
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                            input_var=image_var)

    network = lasagne.layers.DropoutLayer(input_layer, p=0.5)
    psize = lasagne.layers.get_output_shape(network, (250, 1, 28, 28))
    for i, conv_p in enumerate(conv_params):
        drp = 0.2 if (i % 2 == 0) else None
        network = add_conv_layer(network, dropout=0.2, **conv_p)
        nsize = lasagne.layers.get_output_shape(network, (250, 1, 28, 28))
        print('{} -> {}'.format(repr(psize), repr(nsize)))
        psize = nsize

    for dense_params in full_params:
        network = add_full_layer(network, dropout=0.2, **dense_params)

    output_layer = lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, label_var).mean()

    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, label_var).mean()
    test_acc = tt.mean(tt.eq(tt.argmax(test_prediction, axis=1), label_var),
                          dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # maps (image, label) -> loss
    train_fn = theano.function([image_var, label_var], loss, updates=updates)
    # maps (image, label) -> (test loss, test accuracy)
    test_fn = theano.function([image_var, label_var], [test_loss, test_acc])

    out_log = open(logfile, 'w')
    out_log.write('\t'.join(['Epoch', 'train_time', 'loss', 'train_accuracy', 'validation_accuracy']) + '\n')
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_err, train_acc = 0, 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 250, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # a pass to get the train accuracy
        for batch in iterate_minibatches(X_train, y_train, 250, shuffle=False):
            inputs, targets = batch
            err, acc = test_fn(inputs, targets)
            train_acc += acc

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 250, shuffle=False):
            inputs, targets = batch
            err, acc = test_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc/train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        out_log.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, time.time() - start_time, train_err/train_batches, train_acc/train_batches, val_acc/val_batches))
        out_log.flush()

    out_log.close()

if __name__ == '__main__':
    main()

