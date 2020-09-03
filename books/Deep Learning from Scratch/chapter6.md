# 第6章	与学习相关的技巧

## 1.参数更新

1. SGD (Stochastic Gradient Descent)
   $$
   W \leftarrow W - \eta \frac{\partial L_{B_i}}{\partial W}
   $$
   其中$\eta$为学习率，$L_{B_i}$是训练集随机选取的第$i$个Batch。

   > SGD的缺点
   >
   > 给定两个函数分别为：
   > $$
   > f(x, y) = \frac{1}{20}\times x^2 + y^2 \\ 
   > g(x, y) = x^2 + y^2
   > $$
   > 梯度分别为：
   > $$
   > \nabla f = (\frac{1}{10}\times x + 2\times y) \\
   > \nabla g = (2\times x + 2\times y)
   > $$
   > 函数$f, g$均在$(0, 0)$处取最小值。取$x=1, y=1$时：
   > $$
   > -\nabla f_{1, 1} = (-\frac{1}{10}, -2) \\
   > -\nabla g_{1, 1} = (-2, -2)
   > $$
   > 函数$f$在$(1, 1)$处参数更新方向并不指向$(0, 0)$，函数$g$在$(1, 1)$处参数更新方向指向$(0, 0)$。
   >
   > 结论
   >
   > 如果函数的形状非均匀（anisotropic），比如函数$f$，SGD算法搜索路径会非常低效。SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。

2.Momentum
$$
v \leftarrow \alpha v - \eta\frac{\partial L}{\partial W} \\
W \leftarrow W + v
$$
其中$v$对应于物理上的速度，$\alpha$对应于物理上的摩擦力。

3.AdaGrad (Adaptive Gradient)
$$
h \leftarrow h + \frac{\partial L}{\partial W}\bigodot\frac{\partial L}{\partial W} \\
W \leftarrow W - \eta \frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}
$$
4.Adam (AdaGrad + Momentum)



## 6.2 Batch Normalization

Batch Normalization基本思路是调整各层的激活值分布使其拥有适当的广度。

