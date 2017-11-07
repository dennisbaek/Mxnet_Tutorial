import mxnet as mximport mxnet.gluon as gluonimport mxnet.ndarray as ndimport mxnet.autograd as autogradfrom matplotlib import pyplot as pltfrom VGG import get_vggfrom data_preprocessing import data_preprocessingfrom tqdm import *import numpy as npdef artistic_Image(noise_image,image_size):    image = noise_image.reshape((-1,)+image_size)    r,g,b = nd.split(image ,axis=0, num_outputs=3)    #Denormalization by JG    r= nd.multiply(r,0.229)+0.485    g= nd.multiply(g,0.224)+0.456    b= nd.multiply(b,0.225)+0.406    image=nd.concat(r,g,b,dim=0)    '''    matplotlib supports float32 and uint8 data types. For grayscale, matplotlib supports only float32.     If your array data does not meet one of these descriptions, you need to rescale it.    '''    image = nd.transpose(image, axes=(1, 2, 0))    image = nd.clip(image , a_min=0 , a_max=1)    image = nd.multiply(image,255)    image = nd.clip(image, a_min=0, a_max=255).astype('uint8')    plt.imshow(image.asnumpy())    plt.savefig("Artistic Image.png", dpi=200)def neuralstyle(epoch = 100, show_period=100, optimizer = "adam", image_size=(), learning_rate = 0.1, content_image = None, style_image = None, content_a = None, style_b = None,initial_noise_image=None, ctx = mx.gpu(0)):    #1. Data Preprocessing and noise data    content_image , style_image , noise = data_preprocessing(content_image = content_image, style_image = style_image, image_size=image_size ,ctx=ctx)    noise_image = gluon.Parameter('noise', shape=content_image.shape)    noise_image.initialize(ctx=ctx)    #initializing noise image below values    if initial_noise_image=="content_image":        noise_image.set_data(content_image)    elif initial_noise_image=="style_image":        noise_image.set_data(style_image)    else:        noise_image.set_data(noise)    #2. Predefined VGG19 Network    # style : cov1_1 ,cov2_1 ,cov3_1 ,cov4_1 ,cov5_1    select_style=[1, 6, 11, 20, 29]    #content : conv4_2    select_content=[22]    select=select_style+select_content    weighting_factors=nd.divide(nd.ones(shape=np.shape(select_style),ctx=ctx),len(select_style))    vgg19=get_vgg(select=select, num_layers=19 , pretrained=True , batch_norm=False , ctx=ctx)    #Do not use it at all. : The graph of vgg19 should not be drawn. - AssertionError: Wrong number of inputs.    #vgg19.hybridize()    #optimizer    content_loss = gluon.loss.L2Loss()    style_loss = gluon.loss.L2Loss()    trainer = gluon.Trainer([noise_image], optimizer, {"learning_rate" : learning_rate})    #3 learning    for i in tqdm(range(1,epoch+1,1)):        c_loss=nd.array([0,],ctx=ctx)        s_loss=nd.array([0,],ctx=ctx)        with autograd.record(): #Location is very important            content=vgg19(content_image)            noise=vgg19(noise_image.data())            style=vgg19(style_image)            for j,(n,c,s) in enumerate(zip(noise,content,style), start=1):                batch_size, filter , height, width = n.shape                # (1)compute style lose                #using cov1_1 ,cov2_1 ,cov3_1 ,cov4_1 ,cov5_1                if j < len(select):                    #reshape                    n = n.reshape((-1, height*width))                    s = s.reshape((-1, height*width))                    N = filter                    M = height*width                    '''                    The style and noise size can be different                    -> This is because there are only the filter * filter(channel*channel) is left                    as a result of the gram matrix                    '''                    #gram_matrix                    n=nd.dot(n,n,transpose_a=False,transpose_b=True) #(filter, filter)                    s=nd.dot(s,s,transpose_a=False,transpose_b=True) #(filter, filter                    #autograd.record() is not supporting the += operator                    s_loss=s_loss+nd.mean(nd.multiply(nd.divide(style_loss(n,s),2*N*M),weighting_factors[j-1])) #nd.mean((filter,))                    #s_loss = s_loss + nd.mean(nd.multiply(nd.divide(style_loss(n, s), 2 * (N*N) * (M*M)), weighting_factors[j - 1])) # it is too small because of the square of N,M                # (2)compute content lose                #using conv4_2                if j == len(select):                    #reshape                    n = n.reshape((-1, height*width))                    c = c.reshape((-1, height*width))                    '''                    If you do not want the size of the content image and the noise image to be the same,                    you should work with c equal to n. i did not do anything here.                     Therefore, the content size and noise size should be the same.                     '''                    c_loss = content_loss(n,c)                    c_loss = nd.mean(c_loss)            c_loss = nd.multiply(c_loss, content_a)            s_loss = nd.multiply(s_loss, style_b)            loss = c_loss + s_loss        loss.backward()        trainer.step(1,ignore_stale_grad=True)        print(" epoch : {} , cost : {}".format(i, loss.asscalar()))        #saving image        if i%show_period==0:            artistic_Image(noise_image.data(),image_size)if __name__ == "__main__":    content_image = "content/tiger2.jpg"    style_image = "style/rain_princess.jpg"    initial_noise_image = "content_image"    image_size = (512, 512)    neuralstyle(epoch = 1000, show_period=100, optimizer = "adam",image_size=image_size , learning_rate = 0.1, content_image = content_image, style_image = style_image, content_a = 1, style_b = 1000, initial_noise_image=initial_noise_image, ctx = mx.cpu(0))else:    print("Imported")