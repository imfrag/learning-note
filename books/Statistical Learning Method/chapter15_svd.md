# 第15章	奇异值分解

​	奇异值分解（Singular Value Decomposition, SVD）是一种矩阵因子分解方法。任意一个$m\times n$矩阵，都可以表示为三个矩阵的乘积（因子分解）形式，分别是$m$阶正交矩阵、由降序排列的非负的对角线元素组成的$m\times n$矩形对角矩阵和$n$阶正交矩阵，即$A_{m\times n} = U_{m\times m}\sum_{m\times n}V_{n\times n}^{T}$，称为该矩阵的奇异值分解。

​	矩阵的奇异值分解一定存在，但不唯一。奇异值分解可以看做是矩阵数据压缩的一种方法，即用因子分解的方式近似的表示原始矩阵，:question:<u>这种近似是在平方损失意义下的最优近似</u>。

> 15.3.1弗罗贝尼乌斯范数
>
> 矩阵数据压缩：紧奇异值分解对应着无损压缩，截断奇异值分解对应着有损压缩。

## 15.1	奇异值分解的定义与性质

### 15.1.1	定义与定理

#### 定义15.1 奇异值分解

​	矩阵的因子分解：
$$
A_{m\times n} = U_{m\times m}\Sigma_{m\times n}V_{n\times n}^{T}
$$
​	其中
$$
UU^T=I \\
VV^T=I \\
\Sigma=diag(\sigma_1, \sigma_2, \sigma_3 ... \sigma_p) \\
\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq ... \geq \sigma_p \\
p = min(m, n)
$$
​	$\sigma_i$是原始矩阵的奇异值，$U$的列向量称为左奇异向量，$V$的列向量称为右奇异向量。

#### :question:定理15.1 奇异值分解基本原理

### 15.1.2	紧奇异值分解与截断奇异值分解

​	定理15.1给出的奇异值分解$A_{m\times n} = U_{m\times m}\sum_{m\times n}V_{n\times n}^{T}$，又称为完全奇异值分解（Full Singular Value Decomposition）。实际常用的是奇异值分解的紧凑形式和截断形式。紧奇异值分解（Compact Singular Value Decomposition）是与原始矩阵等秩的奇异值分解，截断奇异值分解（Truncated Singular Value Decomposition）是比原始矩阵低秩的奇异值分解。

### ~~15.1.3	几何解释~~

### ~~15.1.4	主要性质~~

## ~~15.2	奇异值分解的计算~~

## 15.3	奇异值分解与矩阵近似

### 15.3.1	弗罗贝尼乌斯范数

​	奇异值分解是一种矩阵近似的方法，这个近似是在弗罗贝尼乌斯范数（Frobenius norm）意义下的近似。矩阵的弗罗贝尼乌斯范数是向量的$L_2$范数的直接推广，对应着机器学习中的平方损失函数。

#### 定义15.4 弗罗贝尼乌斯范数

设矩阵$A\in R^{m\times n}$，定义矩阵$A$的弗罗贝尼乌斯范数为
$$
||A||_F=(\sum_{i=1}^{m}\sum_{j=1}^{n}(a_{ij}^2))^{\frac{1}{2}}
$$

#### 引理15.1	设矩阵$A\in R^{m\times n}$，$A$的奇异值分解为$U\Sigma V^T$，其中$\Sigma=diag(\sigma_1, \sigma_2, ..., \sigma_n)$，则

$$
||A||_F=(\sigma_1^2+\sigma_2^2+...+\sigma_n^2)^{\frac{1}{2}}
$$

> 证明：
>
> 若$Q$是m阶正交矩阵，$\alpha$为m维向量，设$Q\alpha=[\mu_1, \mu_2, ... ,\mu_m]$，则有
> $$
> ||Q\alpha||_F = ||[\mu_1, \mu_2, ... ,\mu_m]||_F \\
> =||\mu_1||_F+||\mu_2||_F+...+||\mu_m||_F \\
> =\mu_1^T\mu_1+\mu_2^T\mu_2+...+\mu_m^T\mu_m \\
> =[\mu_1, \mu_2, ... ,\mu_m]^T[\mu_1, \mu_2, ... ,\mu_m] \\
> =(Q\alpha)^T(Q\alpha) \\
> =\alpha^TQ^TQ\alpha \\
> =\alpha^T\alpha
> =||\alpha||_F
> $$
> 则得
> $$
> ||QA||_F=||Q[a_1, a_2, ..., a_n]||_F \\
> =\sum||Qa_i||_F \\
> =\sum||a_i||_F \\
> =||A||_F
> $$
>
> $$
> ||A||_F=||U\Sigma V^T||_F \\
> =||\Sigma|| \\
> =(\sigma_1^2+\sigma_2^2+...+\sigma_n^2)^{\frac{1}{2}}
> $$
>
> 

### 15.3.2	:question:矩阵的最优近似

### 15.3.3	:question:矩阵的外积展开式