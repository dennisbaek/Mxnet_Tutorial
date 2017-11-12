# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import data_download as dd
import logging
from tqdm import *
import os
logging.basicConfig(level=logging.INFO)

def to2d(img):
    return img.reshape(img.shape[0],784).astype(np.float32)/255.0

def NeuralNet(epoch,batch_size,save_period,load_weights,ctx=mx.gpu(0)):


    (train_lbl_one_hot, train_lbl, train_img) = dd.read_data_from_file('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')
    (test_lbl_one_hot, test_lbl, test_img) = dd.read_data_from_file('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : to2d(train_img)},label={'label' : train_lbl}, batch_size=batch_size, shuffle=True) #training data
    test_iter  = mx.io.NDArrayIter(data={'data' : to2d(test_img)}, label={'label' : test_lbl}, batch_size=batch_size) #test data

    '''neural network'''
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    with mx.name.Prefix("FNN_"):

        # hidden_layer
        affine = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=100)
        hidden = mx.sym.Activation(data=affine, name='relu', act_type="relu")

        # output_layer
        output_affine = mx.sym.FullyConnected(data=hidden, name='fc2', num_hidden=10)

    output=mx.sym.SoftmaxOutput(data=output_affine,label=label)

    # (1) Get the name of the 'argument'
    arg_names = output.list_arguments()
    arg_shapes, output_shapes, aux_shapes = output.infer_shape(data=(batch_size,784))

    # (2) Make space for 'argument' - mutable type - If it is declared as below, it is kept in memory.
    arg_dict = dict(zip(arg_names, [mx.nd.random_normal(loc=0, scale=0.01, shape=shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict= dict(zip(arg_names[1:-1], [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes[1:-1]])) #Exclude input output

    # We visualize the network structure with output size (the batch_size is ignored.)
    shape = {"data": (batch_size,784)}
    graph=mx.viz.plot_network(symbol=output,shape=shape)#The diagram can be found on the Jupiter notebook.
    if epoch==1:
        graph.view()

    #(5) How to get pretrained model from mxnet 'symbol' - VGG19
    if os.path.exists("weights/MNIST_weights-{}.param".format(load_weights)):
        print("MNIST_weights-{}.param exists".format(load_weights))
        pretrained = mx.nd.load("weights/MNIST_weights-{}.param".format(load_weights))

        for name in arg_names:
            if name == "data" or name == "label":
                continue
            else:
                arg_dict[name] = pretrained[name]
    else:
        print("weight initialization")


    '''You only need to bind once.
    With bind, you can change the argument values of args and args_grad. The value of a list or dictionary is mutable.
    See the code below for an approach.'''
    network=output.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req='write')

    #optimizer
    state=[]
    optimizer = mx.optimizer.Adam(learning_rate=0.001)
    '''
    Creates auxiliary state for a given weight.
    Some optimizers require additional states, e.g. as momentum, in addition to gradients in order to update weights. 
    This function creates state for a given weight which will be used in update. This function is called only once for each weight.
    '''
    for shape in arg_shapes[1:-1]:
        state.append(optimizer.create_state(0,mx.nd.zeros(shape=shape,ctx=ctx)))

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        print("epoch : {}".format(i))
        train_iter.reset()
        for batch in train_iter:
            '''
            <very important>
            # mean of [:]  : This sets the contents of the array instead of setting the array to a new value not overwriting the variable.
            '''
            arg_dict["data"][:] = batch.data[0]
            arg_dict["label"][:] = batch.label[0]
            network.forward()
            network.backward()

            for j,name in enumerate(arg_names[1:-1]):
                optimizer.update(0, arg_dict[name] , grad_dict[name] , state[j])

        result = network.outputs[0].argmax(axis=1)
        print('Test batch accuracy : {}%'.format((float(sum(batch.label[0].asnumpy() == result.asnumpy())) / len(result.asnumpy()))*100))

        if not os.path.exists("weights"):
            os.makedirs("weights")

        if i%save_period==0:
            mx.nd.save("weights/MNIST_weights-{}.param".format(i),arg_dict)

    print("#Optimization complete\n")

    #test
    for batch in test_iter:
        arg_dict["data"][:] = batch.data[0].as_in_context(ctx)
        arg_dict["label"][:] = batch.label[0].as_in_context(ctx)
        network.forward(is_train=True)

    result = network.outputs[0].argmax(axis=1)
    print("###########################")
    print('Test batch accuracy : {}%'.format((float(sum(batch.label[0].asnumpy() == result.asnumpy())) / len(result.asnumpy())) * 100))

if __name__ == "__main__":
    print("NeuralNet_starting in main")
    NeuralNet(epoch=100,batch_size=100,save_period=100,load_weights=100,ctx=mx.gpu(0))
else:
    print("NeuralNet_imported")

