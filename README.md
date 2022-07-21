# Quantization
Qauntize weights of DNN to 'N' bits.
# How to Run 
## train 
If you want to train a model with 'N' bits, use the following command:
```
python main.py --dataset cifar10 --arch vgg16_quan --quan_bitwidth 8 --reset_weight --pretrained False 
```
## pretrain 
If you have a pre-trained model, locate the model.pth file in the model folder and run: 
```
python main.py --dataset cifar10 --arch vgg16_quan --quan_bitwidth 8 --reset_weight --pretrained True 
```
This command saves the fixed point weights in the 'save' folder layer by layer. The code test for VGG16 and the layer weights exist in the 'save' folder. The accuracy of fixe point VGG16 on Cifar10 is 84.80.
# Contributors

the code in this repository is based on the following amazing works:
* [https://github.com/elliothe/Neural_Network_Weight_Attack]
