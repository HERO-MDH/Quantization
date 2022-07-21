# Quantization
Qauntize weights of DNN to 'N' bits.
# How to Run 
## train 
If you want train a model with 'N' bits use following command:
```
python main.py --dataset cifar10 --arch vgg16_quan --quan_bitwidth 8 --reset_weight --pretrained False 
```
## pretrain 
If you have pretrained model locate the model.pth file in the model folder and run: 
```
python main.py --dataset cifar10 --arch vgg16_quan --quan_bitwidth 8 --reset_weight --pretrained True 
```
This command save the fixedpoint weights in the save folder layer by layer. 
# Contributors

the code in this repository is based on the following amazing works:
* [https://github.com/elliothe/Neural_Network_Weight_Attack]
