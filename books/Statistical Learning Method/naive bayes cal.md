# 朴素贝叶斯公式推倒

设输入空间$\chi\subseteq\R^n$为n维向量的集合，输出空间为类标记集合$Y=\{C_1, C_2,C_3,\cdots,C_K\}$。给定训练数据集$T=\{(x_1,y_1),(x_2,y_2),(x_3,y_3),\cdots,(x_N,y_N)\}$。

> 对于类标签$Y$
>
> 先验概率$P(Y=C_K)$
>
> 条件概率$P(X=x|Y=C_K)$
>
> 后验概率$P(Y=C_K|X=x)$

$$
指示函数：I(Y=C_i)=\begin{cases}
1, (Y=C_i)\\
0, (Y\neq C_i)\\
\end{cases}
$$

先验概率为
$$
P(Y=C_k) = \frac{\sum_{n=1}^{N}{I(Y_n=C_k)}}{N}, (k = 1, 2, \ldots, K)
$$
条件独立性假设为
$$
P(X=x|Y=C_k) = P(X^{(1)},X^{(2)},\cdots,X^{(n))}|Y=C_k)\\
= \prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=C_k)
$$
后验概率为
$$
P(Y=C_k|X=x)=\frac{P(X=x, Y=C_k)}{P(X=x)}\\
=\frac{P(X=x|Y=C_k)P(Y=C_k)}{P(X=x\cup(Y=C_1\cup Y=C_2\cup \ldots \cup Y=C_K))}\\
=\frac{P(X=x|Y=C_k)P(Y=C_k)}{\sum_{k=1}^{K}P(X=x, Y=C_k)}\\
=\frac{P(X=x|Y=C_k)P(Y=C_k)}{\sum_{k=1}^{K}P(X=x|Y=C_k)P(Y=C_k)}
$$
