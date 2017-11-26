import mxnet as mx
cell_type = "GRU"

if cell_type=="LSTM":
    import LSTM_Cell
    LSTM_Cell.exchange_rate_model(epoch=30, batch_size=256 , save_period=30 , load_period=30 , learning_rate=0.001, ctx=mx.gpu(0))
elif cell_type=="GRU":
    import GRU_Cell
    GRU_Cell.exchange_rate_model(epoch=30, batch_size=256 , save_period=30 , load_period=30 , learning_rate=0.001, ctx=mx.gpu(0))
else :
    print("please write the cell type exactly")
