# 第五章	Logistic回归

### 5.2.2	训练算法：使用梯度上升找到最佳参数

​	对于二分类Logistic回归算法，其条件概率为：
$$
P(Y=1|x;w) = \frac{e^{(wx)}}{1 + e^{(wx)}} = \pi(x) \\
P(Y=0|x;w) = \frac{1}{1 + e^{(wx)}} = 1 - \pi(x)
$$
​	其中$w=[w_1, w_2, ..., w_n, b], x=[x_1, x_2, ..., x_n, 1]$。

​	极大似然估计求参数$w$，即$argmax\prod_i^n\pi(x_i)^{y_i}[1-\pi(x_i)^{(1-y_i)}]$
$$
令L(w) = log(\prod_i^n\pi(x_i)^{y_i}[1-\pi(x_i)]^{(1-y_i)}) \\
= \sum_i^ny_ilog(\pi(x_i))+(1-y_i)log(1-\pi(x_i)) \\
=\sum_i^ny_ilog\frac{\pi(x_i)}{1-\pi(x_i)} + log(1-\pi(x_i)) \\
=\sum_i^ny_iwx_i - log(1 + e^{wx_i})
$$
​	求$w$使得$L(w)$最大，沿参数梯度方向到最大值点，即梯度上升法
$$
\frac{\partial L(w)}{\partial w} = \sum_i^ny_ix_i - x_i\frac{e^{wx_i}}{{1+e^{wx_i}}} \\
=\sum_i^n(y_i - \frac{1}{1+e^{-wx_i}})x_i \\
=\sum_i^n(y_i - sigmoid(wx_i))x_i
$$
