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

BATCH_SIZE = 250
OUTPUT_CHANNELS = 6


def get_args():
    parser = ArgumentParser('run_mnist')
    parser.add_argument('epoch_log', help='The file for the accuracy output by epoch', type=str)
    parser.add_argument('--conv_classif', help='Comma-separated size:stride:filter:pad of classification convolutional layers', type=str, 
        default='5:2:5:1,5:2:24:0')
    parser.add_argument('--conv_recon', help='Comma-separated size:stride:filter:pad of autoencoder convolutional layers', type=str,
        default='6:1:5:0,3:2:15:0,4:1:40:0,4:1:60:0')
    parser.add_argument('--fusion', help='Size of the classif/autoenc fusion latyer', type=int, default=100)
    parser.add_argument('--deep', help='Comma-separated size of FC layers (not including 10-node output). The last entry is the encoding layer.', 
        type=str, default='50')
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)

    args = parser.parse_args()

    classif_convolutions = [dict(zip(['filter_size', 'stride', 'num_filters', 'pad'], map(int, x.split(':'))))
                    for x in args.conv_classif.split(',')]
    recon_convolutions = [dict(zip(['filter_size', 'stride', 'num_filters', 'pad'], map(int, x.split(':'))))
                    for x in args.conv_recon.split(',')]


    # check that the convolutions are even possible
    in_square, min_square = 28, 3  # input is 28 x 28; minimum output of the conv net is 3 x 3

    print('Architecture::Input 28 x 28 x 1')
    print('Architecture::Classification Convolutions')
    for i, conv_params in enumerate(classif_convolutions):
        out_square = 1 + (in_square - conv_params['filter_size'] + 2 * conv_params['pad'])/conv_params['stride']
        if out_square < min_square:
            raise ValueError('Error in convolutional parameters: layer size became < {}'.format(min_square))
        print('Layer {}: {} x {} x {}'.format(i + 1, out_square, out_square, conv_params['num_filters']))
        in_square = out_square

    in_square = 28
    print('Architecture::Reconstruction Convolutions')
    for i, conv_params in enumerate(recon_convolutions):
        out_square = 1 + (in_square - conv_params['filter_size'] + 2 * conv_params['pad'])/conv_params['stride']
        if out_square < min_square:
            raise ValueError('Error in convolutional parameters: layer size became < {}'.format(min_square))
        print('Layer {}: {} x {} x {}'.format(i + 1, out_square, out_square, conv_params['num_filters']))
        in_square = out_square

    conv_out_shape = (BATCH_SIZE, recon_convolutions[-1]['num_filters'], out_square, out_square)

    full = [{'num_units': int(x)} for x in args.deep.split(',')]

    print('Architecture::Fusion')
    print('Layer {}: {}'.format(1 + len(classif_convolutions) + len(recon_convolutions) + 1, args.fusion))  

    print('Architecture::Dense')
    for i, dparams in enumerate(full):
        print('Layer {}: {}'.format(i + len(classif_convolutions) + len(recon_convolutions) + 2, dparams['num_units']))

    return classif_convolutions, recon_convolutions, args.fusion, full, args.epochs, args.epoch_log, conv_out_shape


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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, output='both'):
    assert targets is None or len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if output is 'both':
            yield inputs[excerpt], targets[excerpt], inputs[excerpt]
        elif output is 'labels':
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], inputs[excerpt]


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

def add_deconv_layer(prev_layer, dropout=None, **full_kwargs):
    """\
    Given a previous layer, hook up a deconvolutional layer
    (transposed convolution) with the given arguments

    """
    if 'W' not in full_kwargs:
        full_kwargs['W'] = lasagne.init.GlorotUniform()
    if 'nonlinearity' not in full_kwargs:
        full_kwargs['nonlinearity'] = lasagne.nonlinearities.rectify
    if 'pad' in full_kwargs and 'crop' not in full_kwargs:
        full_kwargs['crop'] = full_kwargs['pad']
        del full_kwargs['pad']

    next_layer = lasagne.layers.TransposedConv2DLayer(prev_layer, **full_kwargs)

    if dropout is not None:
        next_layer = lasagne.layers.DropoutLayer(next_layer, p=dropout)

    return next_layer


def train_autoencoder_layer(params, train_dat, ltype='convolutional', sigma=0.1, batch_size=250, epochs=5):
    assert ltype in {'dense', 'convolutional'}
    params = {k: v for k, v in params.items()}
    if 'W' not in params:
        params['W'] = lasagne.init.GlorotUniform()
    if 'nonlinearity' not in params:
        params['nonlinearity'] = lasagne.nonlinearities.rectify

    ivar = tt.tensor4('input_var')
    ovar = tt.tensor4('output_var')

    network = lasagne.layers.InputLayer(shape=tuple([None] + list(train_dat[0].shape)),
                                        input_var=ivar)
    network = lasagne.layers.GaussianNoiseLayer(network, sigma=sigma)
    if ltype == 'convolutional':
        network = lasagne.layers.Conv2DLayer(network, **params)
        encoder = lasagne.layers.Conv2DLayer(network, filter_size=1, stride=1, num_filters=1)
        params['crop'] = params.get('pad', 0)
        del params['pad']
        network = lasagne.layers.TransposedConv2DLayer(encoder, **params)
        network = lasagne.layers.Conv2DLayer(network, filter_size=1, stride=1, num_filters=1)
    else:
        encoder = lasagne.layers.DenseLayer(network, **params)
        network = lasagne.layers.DenseLayer(encoder, np.prod(train_dat.shape[1:]))
        network = lasagne.layers.ReshapeLayer(network, tuple([batch_size] + list(train_dat.shape[1:])))

    denoised = lasagne.layers.get_output(network)
    recon_loss = lasagne.objectives.squared_error(denoised, ovar).mean()


    encoding = lasagne.layers.get_output(encoder, deterministic=True)
    encode_fn = theano.function([ivar], encoding)

    net_params = lasagne.layers.get_all_params([network], trainable=True)
    updates = lasagne.updates.adadelta(recon_loss, net_params)

    train_fn = theano.function([ivar, ovar], recon_loss, updates=updates, allow_input_downcast=True)

    print("Training autoencoder layer...")
    hdr = ['Epoch', 'time', 'mean_MSE']
    # We iterate over epochs:
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_loss, train_batches = 0, 0
        start_time = time.time()
        for batch in iterate_minibatches(train_dat, None, batch_size, shuffle=True, output='input'):
            inputs, targets = batch
            train_loss += train_fn(inputs, targets)
            train_batches += 1

        args = [epoch, time.time() - start_time, train_loss/train_batches]
        for key, val in zip(hdr, args):
            spacer = '   ' if key != 'Epoch' else ''
            print(spacer + '{}: {}'.format(key, val))

    # now run the training data through the network

    transformed_data = encode_fn(train_dat)

    return network, params['W'], transformed_data  # also return the (pointer to the) weights of interest


def pretrain_autoencoder(train_data, conv_params, dense_size):
    """\
        
    Pretrain an autoencoder (stacked denoising autoencoder)
    and return the resulting network

    """
    # train the layers
    trained_layers = list()
    tdata = train_data
    for conv_param in conv_params:
        layer_train, layer_weights, train_encode = train_autoencoder_layer(conv_param, tdata, ltype='convolutional')
        trained_layers.append((layer_train, layer_weights))
        tdata = train_encode

    if dense_size > 0:
        dense_train, dense_weights, tdata = train_autoencoder_layer({'num_units': dense_size}, tdata, ltype='dense')
        trained_layers.append((dense_train, dense_weights))
    
    return trained_layers


def main():
    classif_conv_params, recon_conv_params, n_fusion, full_params, epochs, logfile, conv_out_shape = get_args()

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
    target_image = tt.tensor4('minist_target')

    RECON_LOSS_ALPHA = 0.95

    classif_net = lasagne.layers.DropoutLayer(input_layer, p=0.2)
    psize = lasagne.layers.get_output_shape(classif_net, (BATCH_SIZE, 1, 28, 28))
    for i, conv_p in enumerate(classif_conv_params):
        drp = 0.2 if (i % 2 == 0) else None
        classif_net = add_conv_layer(classif_net, dropout=drp, **conv_p)
        nsize = lasagne.layers.get_output_shape(classif_net, (BATCH_SIZE, 1, 28, 28))
        print('{} -> {}'.format(psize, nsize))
        psize = nsize

    # pre-train the network
    recon_pretrained = pretrain_autoencoder(X_train, recon_conv_params, 50)
    recon_net = lasagne.layers.DropoutLayer(input_layer, p=0.2)
    psize = lasagne.layers.get_output_shape(recon_net, (BATCH_SIZE, 1, 28, 28))
    for i, conv_p in enumerate(recon_conv_params):
        trained_layer = recon_pretrained[i]
        conv_p_copy = {k: v for k, v in conv_p.items()}
        conv_p_copy['W'] = trained_layer[1]  # trained weights
        drp = 0.2 if (i % 2 == 0) else None
        recon_net = add_conv_layer(recon_net, dropout=drp, **conv_p_copy)
        nsize = lasagne.layers.get_output_shape(recon_net, (BATCH_SIZE, 1, 28, 28))
        print('{} -> {}'.format(psize, nsize))
        psize = nsize

    fused_input = lasagne.layers.ConcatLayer([classif_net, recon_net], axis=1)

    fused_network = add_full_layer(fused_input, num_units=n_fusion, dropout=0.2)

    for dense_params in full_params:
        fused_network = add_full_layer(fused_network, dropout=0.2, **dense_params)

    output_classif = lasagne.layers.DenseLayer(fused_network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    # adapt the encoded layer to the original output shape for reconstruction
    output_deconv = add_full_layer(fused_network, dropout=0.2, num_units=np.prod(conv_out_shape[1:]))
    output_deconv = lasagne.layers.ReshapeLayer(output_deconv, conv_out_shape)
    print(lasagne.layers.get_output_shape(output_deconv, (BATCH_SIZE, 1, 28, 28)))
    for idx in range(len(recon_conv_params))[::-1]:
        params = recon_conv_params[idx]
        params['num_filters'] = recon_conv_params[idx-1]['num_filters'] if idx > 0 else OUTPUT_CHANNELS
        output_deconv = add_deconv_layer(output_deconv, dropout=0.1, **params)
        print(lasagne.layers.get_output_shape(output_deconv, (BATCH_SIZE, 1, 28, 28)))

    output_deconv = add_deconv_layer(output_deconv, dropout=0.1, filter_size=1, stride=1, num_filters=1)  # collapse

    #encoding = lasagne.layers.get_output(fused_network)
    reconstruction, classification = lasagne.layers.get_output([output_deconv, output_classif])
    joint_loss = RECON_LOSS_ALPHA * lasagne.objectives.squared_error(reconstruction, target_image).mean() + \
            (1 - RECON_LOSS_ALPHA) * lasagne.objectives.categorical_crossentropy(classification, label_var).mean()
    

    joint_params = lasagne.layers.get_all_params([output_classif, output_deconv], trainable=True)
    joint_updates = lasagne.updates.adadelta(joint_loss, joint_params)

    joint_train_fn = theano.function([input_layer.input_var, label_var, target_image], joint_loss, updates=joint_updates, allow_input_downcast=True)


    test_prediction, test_reconstruction = lasagne.layers.get_output([output_classif, output_deconv], deterministic=True)
    calc_loss = theano.function([input_layer.input_var, label_var, target_image], 
                [lasagne.objectives.squared_error(test_reconstruction, target_image).mean(),
                 lasagne.objectives.categorical_crossentropy(test_prediction, label_var).mean()])
    test_acc = tt.mean(tt.eq(tt.argmax(test_prediction, axis=1), label_var),
                          dtype=theano.config.floatX)
    test_fn = theano.function([image_var, label_var], test_acc)

    out_log = open(logfile, 'w')
    HDR=['Epoch', 'train_time', 'loss', 'train_MSE', 'train_accuracy', 'validation_MSE', 'validation_accuracy']
    out_log.write('\t'.join(HDR) + '\n')
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        train_rec, train_ent, train_acc, train_batches, train_loss = 0, 0, 0, 0, 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            inputs, targets, target_images = batch
            train_loss += joint_train_fn(inputs, targets, target_images)
            train_batches += 1

        # a pass to get the train accuracy
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=False):
            inputs, targets, target_images = batch
            recon_rsq, label_crossent = calc_loss(inputs, targets, target_images)
            train_rec, train_ent = train_rec + recon_rsq, train_ent + label_crossent
            acc = test_fn(inputs, targets)
            train_acc += acc

        # And a full pass over the validation data:
        val_rec, val_ent, val_acc, val_batches = 0, 0, 0, 0
        for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
            inputs, targets, target_images = batch
            recon_rseq, label_crossent = calc_loss(inputs, targets, target_images)
            acc = test_fn(inputs, targets)
            val_rec, val_ent = recon_rsq + val_rec, label_crossent + val_ent
            val_acc += acc
            val_batches += 1

        args = [epoch, time.time() - start_time, train_loss/train_batches, train_rec/train_batches, train_acc/train_batches, val_rec/val_batches, val_acc/val_batches]
        for key, val in zip(HDR, args):
            spacer = '   ' if key != 'Epoch' else ''
            print(spacer + '{}: {}'.format(key, val))

        out_log.write('\t'.join(map(str, args)) + '\n')
        out_log.flush()

    out_log.close()

if __name__ == '__main__':
    main()

