---
typora-root-url: ..\pic
typora-copy-images-to: ..\pic
---

# Simplifying graph convolutional networks

## Abstract

## Introduction

GNN由什么启发而来？deep learning

由深度学习启发而来的结果？redundant computation

解决？SGC

## Simple Graph Convolution

GCN

提出假设：GNN的学习能力是来自聚合邻居信息，而不是因为非线性层。

SGC

## Experiment

## Conclusion



# ==Notes==

> ==[1] Introduction==
>
> ​	Adding self-loops to the adjacency matrix of graphs improves accuracy.
>
> ![Snipaste_2020-09-20_19-41-13](/Snipaste_2020-09-20_19-41-13.png)
>
> ​	the adjacency matrix A of the given graph represented as
> $$
> A=\begin{bmatrix}
> {0}&{1}&{0}&{0}&{0}&{0}&{0} \\
> {1}&{0}&{1}&{0}&{1}&{0}&{0} \\
> {0}&{1}&{0}&{1}&{0}&{0}&{0} \\
> {0}&{0}&{1}&{0}&{1}&{0}&{0} \\
> {0}&{1}&{0}&{1}&{0}&{1}&{0} \\
> {0}&{0}&{0}&{0}&{1}&{0}&{1} \\
> {0}&{0}&{0}&{0}&{0}&{1}&{0} \\
> \end{bmatrix}, \hat A=A + I
> $$
> ​	K=7
> $$
> A^7=\begin{bmatrix}
> {0}&{71}&{0}&{57}&{0}&{39}&{0} \\
> {71}&{0}&{128}&{0}&{167}&{0}&{39} \\
> {0}&{128}&{0}&{103}&{0}&{71}&{0} \\
> {57}&{0}&{103}&{0}&{135}&{0}&{32} \\
> {0}&{167}&{0}&{135}&{0}&{96}&{0} \\
> {39}&{0}&{71}&{0}&{96}&{0}&{25} \\
> {0}&{39}&{0}&{32}&{0}&{25}&{0} \\
> \end{bmatrix} \\
> \hat A^7=\begin{bmatrix}
> {225}&{477}&{371}&{358}&{455}&{221}&{84} \\
> {447}&{1051}&{835}&{826}&{1056}&{539}&{221} \\
> {371}&{835}&{673}&{670}&{833}&{414}&{161} \\
> {358}&{826}&{670}&{680}&{863}&{455}&{193} \\
> {455}&{1056}&{833}&{863}&{1128}&{642}&{294} \\
> {211}&{539}&{414}&{455}&{642}&{428}&{228} \\
> {84}&{221}&{161}&{193}&{294}&{228}&{134} \\
> \end{bmatrix}
> $$
> ​	according to the K-th power of feature propagation matrix, the matrix without self-loops may loss local information.

> ==[3.1] Preliminaries on Graph Convolutions==
>
> ​	Given spectral filter, $g\in R^n$, and signal defined on the vertices of graph. Thus, the graph convolution operation between singal x（$x\in R^n$） and filter g is
> $$
> g*x = U((U^Tg)\bigodot(U^Tx)) \\
> = U\hat GU^Tx
> $$
> ​	$\hat G$ represented as
> $$
> \hat G = \begin{bmatrix}
> {u_1^Tg}&{0}&{0}&{...}&{0} \\
> {0}&{u_2^Tg}&{0}&{...}&{0} \\
> {0}&{0}&{u_3^Tg}&{...}&{0} \\
> {...}&{...}&{...}&{}&{...} \\
> {0}&{0}&{0}&{...}&{u_n^Tg} \\
> \end{bmatrix}
> $$
> ​	Graph convolutions can be approximated by k-th order polynomials of Laplacians
> $$
> g*x=U\hat GU^T\approx U(\sum_{i=0}^{k}\theta_i\Lambda^i)U^Tx=\sum_{i=0}^{k}\theta_i\Delta^i_{sys}x \\
> \Delta_{sys}^i=D^{-\frac{1}{2}}\Delta D^{-\frac{1}{2}}
> $$
> ​	$k=1, \theta_0=2\theta, \theta_1=-\theta$
> $$
> g*x = (2\theta I - \theta\Delta_{sys})x \\
> =\theta(2I - D^{-\frac{1}{2}}\Delta D^{-\frac{1}{2}})x \\
> =\theta(2I - (I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}))x \\
> =\theta(I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
> $$
> ​	replace the matrix $I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ by a nomalized version $\tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}$， where $\tilde A=A + I$ and consequently $\tilde D = D + I$，dubbed the ==renormalization trick==

> ==[3.2] SGC and Low-Pass Filtering==
> $$
> S_{1-order} = I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \\
> \Delta_{sym} = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \\
> S_{1-order} = 2I - \Delta_{sym}
> $$
> ​	则有$S_{1-order}=2I-U\Lambda U^T=U(2I-\Lambda)U^T=U\hat \Lambda U^T$，令$\lambda_i是\Delta_{sym}$的特征值，则
> $$
> \hat\Lambda = \begin{bmatrix}
> {2-\lambda_1}&{0}&{0}&{...}&{0} \\
> {0}&{2-\lambda_2}&{0}&{...}&{0} \\
> {0}&{0}&{2-\lambda_3}&{...}&{0} \\
> {...}&{...}&{...}&{...}&{...} \\
> {0}&{0}&{0}&{...}&{2-\lambda_n} \\
> \end{bmatrix}
> $$
> ​	则对于SGC，$Y=S^KX\Theta$
> $$
> S^K=(U\hat\Lambda U^T)^K=U\hat\Lambda^KU^T
> $$
> 

> ==[5.1] Citation Networks & Social Networks==
>
> development set：调整参数、选择特征，以及对学习算法做出其决定。
> 

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \tilde h_i^{(k)} \larr \frac{1}{d_i+1}h_i^{(k-1)} + \sum_{j=1}^{n}\frac{a_{ij}}{\sqrt{(d_i+1)(d_j+1)}}h_j^{(k-1)} \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \larr \frac{1}{d_i+1}h_i^{(k-1)} + \frac{1}{\sqrt{d_i+1}}\begin{bmatrix}{a_{i1}}&{a_{i2}}&{...}&{a_{in}}\end{bmatrix}diag({\frac{1}{\sqrt{d_1+1}}}, {\frac{1}{\sqrt{d_2+1}}}, ..., {\frac{1}{\sqrt{d_n+1}}})H^{(k-1)} \\
\larr \frac{1}{d_i+1}h_i^{(k-1)} + \frac{1}{\sqrt{d_i+1}} \alpha_i\tilde D^{-\frac{1}{2}}H^{(k-1)}
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \tilde H^{(k)} \larr \tilde D^{-1}H^{(k-1)} + \tilde D^{-\frac{1}{2}}A\tilde D^{-\frac{1}{2}}H^{(k-1)} \\
\ \ \ \ \ \ \ \ \ \larr \tilde D^{-\frac{1}{2}}(I+A)\tilde D^{-\frac{1}{2}}H^{(k-1)} \\
\larr \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}H^{(k-1)}
$$

> ==[Supplementary Material] C. Propagation choice==
>
> Normalized Adjacency: $\Delta_{sym}=I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}, S_{adj} = D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=I-\Delta_{sym}$
>
> Random Walk Adjacency: $\Delta_{sym}=I-D^{-1}A ,S_{rw} = D^{-1}A=I-\Delta_{sym}$
>
> Augmented Normalized Adjacency: $\tilde \Delta_{sym}=I - \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}},\tilde S_{adj} = \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}=I-\tilde\Delta_{sym}$
>
> Augmented Random Walk Adjaceny: $\tilde\Delta_{rw}=I-\tilde D^{-1}\tilde A, \tilde S_{rw} = \tilde D^{-1}\tilde A=I-\tilde\Delta_{rw}$
>
> First-Order Cheby $\Delta_{sym}=I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}},S_{1-order} = (I+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})=2I-\Delta_{sym}$

