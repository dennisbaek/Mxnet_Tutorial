
>## ***Mxnet?*** 
* ***Flexible and Efficient Library for Deep Learning***
* ***Symbolic programming or Imperative programming***
* ***Mixed programming available*** ***(`Symbolic + imperative`)***
 
<image src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png" width=800 height=200></image>
>## ***Introduction*** 
*   
    It is a tutorial that can be helpful to those `who are new to the MXNET deep-learning framework`
>## ***Official Homepage Tutorial***
*
    The following LINK is a tutorial on the MXNET  official homepage
    * `Link` : [mxnet homapage tutorials](http://mxnet.io/tutorials/index.html)
>## ***Let's begin with***
* Required library and very simple code
```python
import mxnet as mx
import numpy as np

out=mx.nd.ones((3,3),mx.gpu(0))
print(mx.asnumpy(out))
```
* The below code is the result of executing the above code
```
<NDArray 3x3 @gpu(0)>
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
```      
>## ***Topic 1 : Symbolic Programming***
* ### ***Neural Networks basic with <Symbol.API + Module.API>***
    * [***Fully Connected Neural Network with LogisticRegressionOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with_LogisticRegressionOutput)

    * [***Fully Connected Neural Network with SoftmaxOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with_softmax)
    
    * [***Fully Connected Neural Network with SoftmaxOutput*** *(flexible)* : ***Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Fully%20Connected%20Neural%20Network%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Convolutional Neural Networks with SoftmaxOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput)

    * [***Convolutional Neural Networks with SoftmaxOutput*** *(flexible)* : ***Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Convolutional%20Neural%20Networks%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Recurrent Neural Networks with SoftmaxOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20with%20SoftmaxOutput)
    
    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput)

    * [***Recurrent Neural Networks + LSTM with SoftmaxOutput*** *(flexible)* : ***Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20LSTM%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module)) 
    
    * [***Recurrent Neural Networks + GRU with SoftmaxOutput : Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20GRU%20with%20SoftmaxOutput)

    * [***Recurrent Neural Networks + GRU with SoftmaxOutput*** *(flexible)* : ***Classifying the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Recurrent%20Neural%20Networks%20%2B%20GRU%20with%20SoftmaxOutput(flexible%20to%20use%20the%20module))

    * [***Autoencoder Neural Networks with logisticRegressionOutput : Compressing the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput)

    * [***Autoencoder Neural Networks with logisticRegressionOutput*** *(flexible)* : ***Compressing the MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/basic/Autoencoder%20Neural%20Networks%20with%20logisticRegressionOutput(flexible%20to%20use%20the%20module))

* ### ***Neural Networks with visualization***
    * [***mxnet with graphviz library***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/visualization/visualization)
        ```cmd
        <linux>
        pip install graphviz(in anaconda Command Prompt) 

        <window>
        1. download 'graphviz-2.38.msi' at 'http://www.graphviz.org/Download_windows.php'
        2. Install to 'C:\Program Files (x86)\'
        3. add 'C:\Program Files (x86)\Graphviz2.38\bin' to the environment variable PATH
        ```
        ```python
        '''<sample code>'''  
        import mxnet as mx  

        '''neural network'''
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        # first_hidden_layer
        affine1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=50)
        hidden1 = mx.sym.Activation(data=affine1, name='sigmoid1', act_type="sigmoid")

        # two_hidden_layer
        affine2 = mx.sym.FullyConnected(data=hidden1, name='fc2', num_hidden=50)
        hidden2 = mx.sym.Activation(data=affine2, name='sigmoid2', act_type="sigmoid")

        # output_layer
        output_affine = mx.sym.FullyConnected(data=hidden2, name='fc3', num_hidden=10)
        output=mx.sym.SoftmaxOutput(data=output_affine,label=label)

        # Create network graph
        graph=mx.viz.plot_network(symbol=output)
        graph.view() # Show graphs and save them in pdf format.
        ```
    * [***mxnet with tensorboard Only available on Linux - currently only works in Python 2.7***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/visualization/tensorboard-linux)
        
        ```python
        pip install tensorboard   
        ```
        ```python
        '''Issue'''
        The '80's line of the tensorboard file in the path '/home/user/anaconda2/bin' should be modified as shown below.
        ```
        ```python
        <code>
        for mod in package_path:
            module_space = mod + '/tensorboard/tensorboard' + '.runfiles'
            if os.path.isdir(module_space):
                return module_space
        ```
        * If you want to see the results immediately,`write the following script in the terminal window` where the event file exists.
        
            * `tensorboard --logdir=tensorboard --logdir=./ --port=6006`

* ### ***Neural Networks Applications***
    * [***Predicting lotto numbers in regression analysis***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Predicting%20lotto%20numbers%20in%20regression%20analysis%20using%20mxnet)

    * [***Generative Adversarial Networks with fullyConnected Neural Network : using MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Generative%20Adversarial%20Network%20with%20FullyConnected%20Neural%20Network)

    * [***Deep Convolution Generative Adversarial Network : using ImageNet , CIFAR10 , MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/Deep%20Convolution%20Generative%20Adversarial%20Network)
        ```cmd
        <Code execution example>  
        python main.py --state --epoch 100 --noise_size 100 --batch_size 200 --save_period 100 --dataset CIFAR10 --load_weights 100
        ```
    * [***Neural Style - using Symbol.API only. - It is being modified***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/applications/NeuralStyle)
        ```cmd
        to configure a network of flexible structure, use 'bind function' of Symbol.API and 'optimizer class'. - See the code for more information!!!
        ```

* ### ***How to Create New Operators for Symbolic Operator? - Advanced***
    * #### [References Page : It is insufficient.](https://mxnet.incubator.apache.org/how_to/new_op.html) 
        
        * [***Customized LogisticRegressionOutput Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20LogisticRegressionOutput)

        * [***Customized SoftmaxOutput Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20SoftmaxOutput)

        * [***Customized SoftmaxOutput + Customized Activation Layer***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Symbol/advanced/Fully%20Connected%20Neural%20Network%20with%20Custom%20Activation)

>## ***Topic 2 : Imperative Programming***

* ### ***Neural Networks With <NDArray.API + Autograd Package + Optimizer Class>***

    ```python
    The NDArray API is different from mxnet-symbolic coding.
    It is imperactive coding and focuses on NDArray of mxnet.
    ```

    * #### The following LINK is a tutorial on the [gluon page](http://gluon.mxnet.io/index.html)  official homepage

        * [***Multiclass logistic regression : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Multiclass_logistic_regression_with_NDArray)

        * [***Fully Neural Network : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Fully_Neural_Network_with_NDArray)

        * [***Convolution Neural Network : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_with_NDArray)

        * [***Convolution Neural Network With Batch Normalization(First Method) : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_BN1_with_NDArray)

        * [***Convolution Neural Network With Batch Normalization(second Method) : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_BN2_with_NDArray)

        * [***Convolution Neural Network With Builtin Batch Normalization : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Neural_Network_Using_builtin_BN_with_NDArray)

        * [***Autoencoder Neural Networks : Using the MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Autoencoder_Neural_Network_with_NDArray)

        * [***Convolution Autoencoder Neural Networks : Using the MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/NDArray/Convolution_Autoencoder_Neural_Network_with_NDArray)

        * [***Recurrent Neural Network : Classifying the MNIST and Fashion MNIST data - not yet***]()

* ### ***Neural Networks With <NDArray.API + Gluon Package + Autograd Package>***

    ```python
    The Gluon package is different from mxnet-symbolic coding.
    It is imperactive coding and focuses on mxnet with NDArray and Gluon.
    ```

    * #### The following LINK is a tutorial on the [gluon page](http://gluon.mxnet.io/index.html)  official homepage

        * ***Using nn.Sequential when writing your `High-Level Code`***

            * [***Multiclass logistic regression : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Multiclass_logistic_regression_with_Gluon)

            * [***Fully Neural Network : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Fully_Neural_Network_with_Gluon)

            * [***Convolution Neural Network : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_with_Gluon)

            * [***Convolution Neural Network with Batch Normalization : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_BN_with_Gluon)

            * [***Autoencoder Neural Networks : Using the MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Autoencoder_Neural_Network_with_Gluon)

            * [***Convolution Autoencoder Neural Networks : Using the MNIST and Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Autoencoder_Neural_Network_with_Gluon)

            * [***Recurrent Neural Network : Classifying the MNIST and Fashion MNIST data - not yet***]()

        * ***Using Block when Designing a `Custom Layer` - Flexible use of Gluon***

            * [***Convolution Neural Network with Block or HybridBlock : Classifying the MNIST , CIFAR10 , Fashion_MNIST data***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Convolution_Neural_Network_with_Block)

                * [More information on `block` and `hybridblock`](http://thestraightdope.mxnet.io/chapter07_distributed-learning/hybridize.html) 
        
* ### ***Neural Networks Applications***

    * [***Predicting lotto numbers in regression analysis***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/Predicting_lotto_numbers_in_regression_analysis_using_Gluon)
    
    * [***Neural Style***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/NeuralStyle)
        ```cmd
        To configure a network of flexible structure, You should be able to set the object you want to differentiate. - See the code for more information!!!
        ```

    * [***Generative Adversarial Networks with CIFAR10 , FashionMNIST***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/DCGAN_with_Gluon)

    * [***Generative Adversarial Networks Targeting with CIFAR10 , MNIST***](https://github.com/JONGGON/Mxnet_Tutorial/tree/master/Gluon/DCGAN_target_with_Gluon)

    * [***Pixel to Pixel Generative Adversarial Networks - not yet***]()

    * I plan to continue various projects in the future.!!!!!


>## ***Development environment***
* os : ```window 10.1 64bit``` and ```Ubuntu linux 16.04.2 LTS only for tensorboard``` 
* python version(`3.6.2`) : `anaconda3 4.4.0` 
* IDE : `pycharm Community Edition 2017.2.2 or visual studio code`
    
>## ***Dependencies*** 
* mxnet-0.12.0
* numpy-1.12.1, matplotlib-2.0.2 , tensorboard-1.0.0a7(linux) , graphviz -> (`Visualization`)
* tqdm -> (`Extensible Progress Meter`)
* opencv-3.3.0.10 , struct , gzip , os , glob , threading -> (`Data preprocessing`)
* Pickle -> (`Data save and restore`)
* logging -> (`Observation during learning`)
* argparse -> (`Command line input from user`)
* urllib , requests -> (`Web crawling`) 
>## ***Author*** 
[JONGGON KIM](https://github.com/JONGGON)
