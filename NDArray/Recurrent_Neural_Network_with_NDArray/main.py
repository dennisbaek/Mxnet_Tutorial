import mxnet as mx
#implementation
cell_type = "RNN"
#dataset = MNIST or CIFAR10 or FashionMNIST
if cell_type=="RNN":
    import RNN
    RNN.RNN(epoch=1, batch_size=256 , save_period=100 , load_period=100 , learning_rate=0.001, ctx=mx.gpu(0))
'''
elif cell_type=="LSTM":
    import LSTM
    LSTM.LSTM(epoch=10, batch_size=256 , save_period=10 , load_period=10 , learning_rate=0.001, ctx=mx.gpu(0))
elif cell_type=="GRU":
    import RNN
    GRU.GRU(epoch=10, batch_size=256 , save_period=10 , load_period=10 , learning_rate=0.001, ctx=mx.gpu(0))
'''
'''else :
    print("please write the cell type exactly")'''
