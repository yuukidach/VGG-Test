# Implement VGG-19 & VGG-34 with PyTorch

| Environment | Version |
| :---: | :---: |
| Computer| Lenovo Y7000P i7 1660Ti |
| Python | 3.8.5 |
| PyTorch | 1.6.0 |
| With GPU| True |
| Test Dataset| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |

The specific description of the task can be seen in this [doc](Deep&#32;Learning&#32;Application&#32;in&#32;Computer&#32;Vision.pdf)



## Test result

Train and test VGG-19 after 100 epochs (start from epoch 0):

![](./res/vgg19-100epoch.png)

Train and test VGG-34 after 100 epochs (start from epoch 0):

![](./res/vgg34-100epoch.png)

## Conclusion

After training 100 epochs, VGG-19 performs much better than VGG-34 in CIFAR-10 dataset, while VGG-34 runs faster.

This result accurred might because the deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error 