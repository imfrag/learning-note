# 《Graph Neural Architecture Search》

>==[3.3] Search Algorithm==
>
>1.Exponential Moving Average (EMA)
>
>​	EMA是指数移动平均值，是一种趋向类指标，指数移动平均值是以指数式递减加权的移动平均。
>
>定义式：对序列${x_n}$定义其截至第n项的周期为N的指数移动平均为
>$$
>EMA_A(x_n)=\frac{2}{N=1}\sum_{k=0}^{\infin}(\frac{N-1}{N+1})^kx_{n-k}
>$$
>​	由于$x_1$之前没有数据，补充定义$x_0=x_{-1}=x_{-2}=...=x_1$。
>
>递推式：
>$$
>EMA_N(x_n)=\frac{2x_n+(N-1)EMA_N(x_{n-1})}{N+1}
>$$
>