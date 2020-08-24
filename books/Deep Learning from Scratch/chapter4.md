1. [P92] 为什么神经网络使用损失函数优化模型，而不是精度？

   如果采用精度作为优化模型的指标，即：
   $$
   Acc = \frac{1}{N}\sum_{1}^{N}S(\hat{y}, y)
   $$
   其中，$N$为样本数；$\hat{y}, y$分别表示预测值和真实值；$S$为阶跃函数：
   $$
   S(\hat{y}, y) = \begin{cases}
   1,\hat{y}与y符合条件 \\
   0, \hat{y}与y不符合条件\\
   \end{cases}
   $$
   预测值可表示为：
   $$
   \hat{y} = f_{model}(x;\theta)
   $$
   $\theta$为模型参数。

   $Acc$对参数$\theta$求导，此时，若导数大于0，则增大$\theta$；反之，减小$\theta$。但是对于精度使用的阶跃函数，其导数处处为0。
   
   如果采用损失函数作为优化模型的指标，即：
   $$
   Loss=\frac{1}{2}\sum_{i=1}^{N}{(\hat y_i - y_i)^2} \\
   \frac{\partial Loss}{\partial \hat y} = \sum_{i=1}^{N}{(\hat y_i - y_i)}
   $$
   由上式可得，损失函数关于$\hat y$并非处处为0。

