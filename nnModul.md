# Pytorch系列之——模型创建与nn.Module
------
## 模型创建
对于模型模块，首先分为两大部分分别是模型的创建和模型权值的初始化。其中在模型创建模块，有分为构建网络层和拼接网络层这两个子模块，我们通过构建类似卷积层、池化层和激活函数层等子模块，再根据一定的顺序或拓扑结构将这些子模块拼接起来，最后就可以构建成类似LeNet、AlexNet和ResNet等深度神经网络。而这些都是基于pytorch中的nn.Module模块来实现的，这是根本也是关键之处。<br>
在人民币二分类实验中，已经实现过了一个神经网络LeNet，这里进行回顾：<br>
![](https://img-blog.csdnimg.cn/20200904093615115.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
计算图前面介绍张量的时候也涉及过，计算图中有两个主要的元素，一个是节点一个是边，节点表示数据，边就是运算。LeNet的模型运算过程是比较复杂的，首先需要对数据（图像）进行卷积操作得到28 x 28 x 6的图像，之后再进行池化处理得到14 x 14 x 6的图像，这样不断地向前向传播，最终得到输出概率。<br>
    
 从这里我们知道，构建模型包含两个要素：构建子模块和拼接子模块。下面通过代码来学习模型创建的步骤：<br>
 ![](https://img-blog.csdnimg.cn/20200904094706357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
     
依然是在LeNet模型构建处设置断点进行Debug调试，stepinto到LeNet这个类中进一步观察它具体的实现步骤以及怎样去构建一个模型：<br>
 ![](https://img-blog.csdnimg.cn/20200904094940786.png#pic_center)
 可以看到，在LeNet这个实现类中，在__init__()函数中首先调用了LeNet父类的初始化函数，再根据nn模块创建了一系列的网络子模块，因此在模型构建的二要素中，构建子模块是在__init__()这个函数里完成的。但是什么时候去拼接这些子模块呢？<br>
 ![](https://img-blog.csdnimg.cn/20200904095421190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
     
我们依然step into光标所在处：<br?
![](https://img-blog.csdnimg.cn/20200904095507669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
程序进入到了module.py文件中的__call__()函数，由于我们想要探讨的是子模块是怎么拼接的，也就是模型的怎么forward的，所以目前只需要关注__call__()函数中的forward()的使用，在forward函数处执行Run on Cursor并继续stepinto，可以看到程序跳转到了LeNet.py文件中的forward()函数：<br>
![](https://img-blog.csdnimg.cn/20200904095904963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
在forward()函数中实现了模型的每一层的计算，上一层的输出作为下一层的输入，逐层传递，不断向前传播最后可以得到一个输出分类结果out（分类概率向量）。<br>
因此，模型创建过程中，构建子模块就是在LeNet类中的__init__()初始化方法中进行的，拼接子模块就是在LeNet类中的forward()函数中进行的，也就是前向传播的一个实现。<br>

-----
## nn.Module
我们所有的模型和网络层都是继承于nn.Module这个类的，所以非常有必要去了解nn.Module这个类。<br>
![](https://img-blog.csdnimg.cn/20200904100711379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
nn.Module模块是属于torch.nn模块的一个子模块，在torch.nn模块下还包含其他三个子模块，nn.Parameter这个用来表示可学习的参数，比如权重和偏差；nn.functional封装了像卷积、池化、激活函数等函数的具体实现；nn.init模块用来是实现模型参数初始化的方法。<br>
在nn.Module中，有8个重要的有序字典：<br>
![](https://img-blog.csdnimg.cn/20200904101048981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
* parameters:存储管理nn.Parameter类
* modules:存储管理nn.Module类
* buffers:存储管理缓冲属性，如BN层中的running_mean
* ***_hooks:存储管理钩子函数
    
重点主要是parameters和modules这两个属性。那么继续设置断点Debug…
![](https://img-blog.csdnimg.cn/20200904101555318.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
     
可以看到LeNet是继承自nn.Module的，init()的第一行是实现一个父类的函数调用的功能。由于LeNet的父类是nn.Module所以它会调用nn.Module的初始化函数__init__()。stepinto到nn.Module的__init__()函数一探究竟：
![](https://img-blog.csdnimg.cn/20200904101843566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
在__init__()方法中实现了8个有序字典的初始化过程，这里着重关注parameters和modules，之后我们stepout这个函数，可以看到已经有了这8个有序字典，但这8个有序字典都是空的：
![](https://img-blog.csdnimg.cn/20200904102120641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
接着我们开始构建子模块，进入stepinto第一个卷积层进行观察：
![](https://img-blog.csdnimg.cn/20200904102327911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
可以看到这里还是调用了Conv2d的父类__init__()方法，接着stepinto它的父类一探究竟：
![](https://img-blog.csdnimg.cn/20200904102516776.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
可以看到ConvNd也是继承自Module的，所以conv2d继承ConvNd,ConvNd继承自Module，所以conv2d它还是一个Module。再次stepinto，可以看到依然是那8个有序字典的初始化：
![](https://img-blog.csdnimg.cn/20200904102820425.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
stepout返回，这时第一个卷积层已经初始化完毕。构建完毕之后LeNet会记录这个卷积层的初始化过程，由于网络层是一个module，所以相关记录会在modules这个字典当中，
![](https://img-blog.csdnimg.cn/20200904103049262.png#pic_center)
可以看到modules字典里已经存在了一个卷积层，它的名称是’conv1’，由于conv2d也是一个module所以它肯定也会有那8个有序属性。对于第二行卷积层依然采用stepinto stepout的方式进行观察，但发现modules当中现在还没有conv2这个属性，这是因为stepinto stepout之后我们只是实现了nn.Conv2d的一个实例化，还没有赋值到LeNet类属性conv2当中。只是构建了网络层，下一步才是赋值到conv2中。
在Module里面有个机制，它会拦截所有的类属性赋值操作。在第二行卷积层处stepinto，程序会跳转到module.py文件中的__setattr__()函数：
![](https://img-blog.csdnimg.cn/20200904104333646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
这个函数的功能就是会拦截所有类属性的赋值，我们在构建万nn.Conv2d第二个卷积层后还没有进行赋值给conv2属性的操作，就被__setattr__()这个函数给拦截了。
![](https://img-blog.csdnimg.cn/20200904104557976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
这个函数主要是先判断这个传入的value的类型，如果value的类型是parameter的话，它就会存储到parameters这个有序字典当中，
![](https://img-blog.csdnimg.cn/20200904104751972.png#pic_center)

可以看到这时传入的value是个Module，因为它是Conv2d所以它会被存到modules这个有序字典当中。
![](https://img-blog.csdnimg.cn/20200904104935986.png#pic_center)
这里的modules[name]中的name就是我们的类属性’conv2’。这时候我们stepout返回，即可看到modules字典中已经有了conv2这个类属性了：
![](https://img-blog.csdnimg.cn/20200904105050878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Rlc3BhY2l0bzEwMDY=,size_16,color_FFFFFF,t_70#pic_center)
每一次网络都会像上述一样计算和存储。以上就是nn.Module属性构建的机制。
## 总计
* 一个module可以包含多个子module
* 一个module相当于一个运算，必须实现forward（）函数
* 每个module都有8个有序字典管理它的属性
