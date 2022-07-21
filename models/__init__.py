from .net_mnist import small_cnn

############# FC for MNIST ####################
from .vanilla_models.vanilla_FC_mnist import FC_5
from .quan_FC_mnist import FC_3_quan
############# LENET for MNIST ####################
from .vanilla_models.vanilla_lenet_mnist import  lenet_5
from .quan_lenet_mnist import lenet_5_quan
############# ResNet for ImmageNet ####################
# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18
from .quan_resnet_imagenet import resnet18_quan, resnet34_quan
from .quan_alexnet_imagenet import alexnet_quan


############## ResNet for CIFAR-10 ###########
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
from .quan_resnet_cifar import resnet20_quan
from .bin_resnet_cifar import resnet20_bin

############## VGG for CIFAR #############

from .vanilla_models.vanilla_vgg_cifar import vgg11_bn, vgg11
from .quan_vgg_cifar import vgg11_bn_quan, vgg11_quan,vgg16_quan
from .bin_vgg_cifar import vgg11_bn_bin


############# Mobilenet for ImageNet #######
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2

from .quan_mobilenet_imagenet import mobilenet_v2_quan