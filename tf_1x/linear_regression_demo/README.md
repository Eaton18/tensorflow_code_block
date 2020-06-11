# Linear Regression Demo

## 梯度爆炸
在极端情况下，权重的值变得非常大（由于学习率设置不合理），以至于溢出，导致NaN值  
如何解决梯度爆炸问题（深度神经网络当中更容易出现）：
1. 重新设计网络
2. 调整学习率
3. 使用梯度截断（在训练过程中检查和限制梯度的大小）
4. 使用激活函数

## tensorboard
```shell script
# tensorboard summary 保存在/linear_regression_demo/summary/
tensorboard --logdir ./summary/
```

参考：[TensorFlow——实现简单的线性回归](https://blog.csdn.net/weixin_42008209/article/details/82715202)
