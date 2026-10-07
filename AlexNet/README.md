# AlexNet
PyTorch implementation of AlexNet from scratch, based on the paper 
*ImageNet Classification with Deep Convolutional Neural Networks*.

# Architecture
5 convolutional layers (96 -> 256 -> 384 -> 384 -> 256 filters)
ReLU, Local Response Normalization and overlapping Max Pooling
3 fully connected layers (4096 -> 4096 -> number of classes) with Dropout
