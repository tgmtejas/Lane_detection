# Lane detection
Goal : </br>
The overall goal of the project is to generate segmented image for the drivable surface of the road.

Dataset :</br>
Data set has been simulated from Carla simulator. There are 1000 images with different envirinments like sunlight, rainy and overcast weather. </br>

Data Pre-processing :</br>
Label images are encoded images with 13 different classes as per Carla document. I was interested in only lane class, so lane class pixels are kept as zero and all other classes set to 1. </br>

Data Augmentation :</br>
1000 images are augmented to 3000 images using Keras's ImageDataGenerator function with help of rotation and scaling. </br>

Algorithms:
1) Global Convolution Network:
Large kernels can be useful for localisation of object. But kernel size increased, number of parameters also increases. With square convolutions like 3x3, 7x7, etc, we can’t make them too big without taking a massive speed and memory consumption hit. One dimensional kernels on the otherhand scale much more efficiently and we can make them quite big without slowing down the network too much. In addition, the paper does use small 3x3 convolutions with a low filter count for efficient refinement of anything the one dimensional convs might miss.
https://arxiv.org/pdf/1703.02719.pdf </br>

2) DeepLabV3+: </br>
This paper uses encoder decoder architecture along with pyramidal concatenation at end of encoder. Instead of using different size kernels in pyramidal unit, it uses different dilation rate on same kernel. This dilated convolution or astrous convolution could be helpful to localise large objects in image without increasing size of kernel. Decoder is simple bilinear upsampler but instead of upsampling in straight with factor of 16 like previous versions of deeplabs, it uses two step 4 factor upsamplers. First upsamples output is concatenated with 2nd pooling output in encoder to detect middle level features in image and then it is upsampled by next upsampler to get final segmented image. https://arxiv.org/pdf/1802.02611.pdf
