# 1 支持向量机
&emsp;&emsp;对于分类问题，训练数据集$D={(x_1,y_1),...,(x_m,y_m)}, y_i∈{-1,+1}$，可能存在很多将数据分类的超平面，而如何找到最佳的一个这就是支持向量机要解决的问题。
## 1.1 支持向量
&emsp;&emsp;见划分超平面用线性方程表示为：
$$
\textbf{w}^T\textbf{x}+b=0
$$
&emsp;&emsp;则样本空间任意点$x$到该超平面的距离为:
$$
r=\frac{\text{w}^T\textbf{x}+b}{||\textbf{w}||}
$$
&emsp;&emsp;假设对所有的训练集都正确可分且存在如下条件:
$$
\begin{aligned}
&\textbf{w}^T\textbf{x}_i+b≥gap,y_i=1;\\
&\textbf{w}^T\textbf{x}_i+b≤-gap,y_i=-1;\\
\end{aligned}
$$
&emsp;&emsp;定义训练集上使得上面的可分条件等号成立的点，即落在可分超平面上的点位支持向量(support vector)，两个不同类别的支持向量到超平面的距离之和为：
$$
γ=\frac{2}{||w||}
$$
&emsp;&emsp;$γ$称之为间隔。
![](sv.png)

&emsp;&emsp;欲找到最大间隔的划分超平面也就是使得:
$$
\begin{aligned}
& \mathop{max}\limits_{w,b} \frac{2}{||\textbf{w}||}\\
& s.t. y_i(\textbf{w}^T\textbf{x}_i+b)≥1, i=1,2,...,m \\
⇒& \mathop{max}\limits_{w,b}\frac{1}{2}||\textbf{w}||^w \\
& s.t. y_i(\textbf{w}^T\textbf{x}_i+b)≥1, i=1,2,...,m
\end{aligned}
$$

## 1.2 求解
&emsp;&emsp;可以将上面的问题通过拉格朗日乘子法得到对偶问题，对每个拉格朗日乘子添加约束$\alpha_i≥0$，则问题的拉格朗日函数可写为：
$$
L(\textbf{w},b,\textbf{α})=\frac{1}{2}||\textbf{w}||^2+\sum_{i=1}^m{α_i(1-y_i(\textbf{w}^T\textbf{x}_i+b))}
$$
&emsp;&emsp;分别对$w,b$求偏导:
$$
\begin{aligned}
    \textbf{w}=∑_{i=1}^m α_i y_i\textbf{x}_i\\
    -=∑_{i=1}^m α_i y_i
\end{aligned}
$$
&emsp;&emsp;带入得到原问题的对偶问题：
$$
\begin{aligned}
   &\mathop{max}\limits_α∑_{i=1}^m α_i-\frac{1}{2}∑_{i=1}^m∑_{j=1}^mα_iα_jy_iy_j\textbf{x}_i^T\textbf{x}_j\\
s.t. &∑_{i=1}^mα_iy_i=0,\\
&α_i≥0, i=1,2,...,m 
\end{aligned}
$$
&emsp;&emsp;解出$\textbf{α}$得到：
$$
f(\textbf{x})=∑_{i=1}^mα_iy_i\textbf{x}^T_i\textbf{x}+b
$$
&emsp;&emsp;需要注意的是$α_i$的解需要满足KKT条件：
$$
\left\{\begin{array}{l}
{\alpha_{i} \geqslant 0} \\
{y_{i} f\left(x_{i}\right)-1 \geqslant 0} \\
{\alpha_{i}\left(y_{i} f\left(x_{i}\right)-1\right)=0}
\end{array}\right.
$$
&emsp;&emsp;对于上式$α_i=0$或者$y_if(\textbf{x}_i)=1$成立，如果$α_i=0$则该样本不会对结果有影响；若$y_if(\textbf{x}_i)=1$则该样本就是支持向量。

## 1.3 SMO
&emsp;&emsp;上述问题的求解本身和样本量挂钩，SMO是一个解决上述问题的高效算法。
&emsp;&emsp;SMO 的基本思路是先固定$α_i$之外的所有参数，然后求向上的极值。由于存在约束$∑_{i=1}^mα_iy_i=0$，若固定$α_i$之外的其他变量,则$α_i$可由其他变量导出。SMO每次选择两个变量$α_i,α_j$，并固定其他参数。这样，在参数初始化后，SMO不断执行如下两个步骤直至收敛:
- 选取一对需要更新的变量$α_i$和$α_j$;
- 固定两个变量之外的其他参数，求解更新后的解。

&emsp;&emsp;注意到只需选取的$α_i$和$α_j$中有一个不满足KKT条件，目标函数就会在选代后减小。直观来看， KKT 条件违背的程度越大，则变量更新后可能导致的目标函数值减幅越大。于是，SMO先选取违背KKT条件程度最大的变量。第二个变量应选择一个使目标函数值减小最快的变量，但由于比较各变量所对应的目标函数值减幅的复杂度过高，因此 SMO 采用了一个启发式:使选取的两变量所对应样本之间的问隔最大. 种直观的解释是，这样的两个变量有很大的差别，与对两个相似的变量进行更新相比，对它们进行更新会带给目标函数值更大的变化。
&emsp;&emsp;而$b$的求值是通过所有支持向量$S=\left\{i | \alpha_{i}>0, i=1,2, \ldots, m\right\}$求解的平均值得到：
$$
b=\frac{1}{|S|} \sum_{s \in S}\left(y_{s}-\sum_{i \in S} \alpha_{i} y_{i} x_{i}^{\mathrm{T}} x_{s}\right)
$$

## 1.4 核函数
&emsp;&emsp;对于非线性可分问题可以通过核函数映射将非线性可分问题转换成线性可分问题，即将问题转换成（$\phi$为映射函数）：
$$
f(\textbf{x})=\textbf{x}^Tφ(\textbf{x})+b
$$
&emsp;&emsp;同样的上面的目标问题就变成了:
$$
\begin{aligned}
&\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}\\
&\text { s.t. } y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \phi\left(\boldsymbol{x}_{i}\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m
\end{aligned}
$$
&emsp;&emsp;对偶问题：
$$
\begin{aligned}
   &\mathop{max}\limits_α∑_{i=1}^m α_i-\frac{1}{2}∑_{i=1}^m∑_{j=1}^mα_iα_jy_iy_jφ(\textbf{x}_i)^Tφ(\textbf{x}_j)\\
s.t. &∑_{i=1}^mα_iy_i=0,\\
&α_i≥0, i=1,2,...,m 
\end{aligned}
$$
&emsp;&emsp;为了避开求解特征空间映射的内积，则定义：
$$
\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\left\langle\phi\left(\boldsymbol{x}_{i}\right), \phi\left(\boldsymbol{x}_{j}\right)\right\rangle=\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right)
$$
&emsp;&emsp;问题转换成：
$$
\begin{array}{ll}
{\max _{\alpha}} & {\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)} \\
{\text { s.t. }} & {\sum_{i=1}^{m} \alpha_{i} y_{i}=0} \\
{} & {\alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m}
\end{array}
$$
&emsp;&emsp;求解为：
$$
\begin{aligned}
f(\boldsymbol{x}) &=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b \\
&=\sum_{i=1}^{m} \alpha_{i} y_{i} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi(\boldsymbol{x})+b \\
&=\sum_{i=1}^{m} \alpha_{i} y_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)+b
\end{aligned}
$$
&emsp;&emsp;上式中$κ$便是核函数。
&emsp;&emsp;核函数：令$\mathcal{X}为输入空间，$κ(,)$定义在$\mathcal{X} \times \mathcal{X}$上的对称函数，则$κ$是核函数当且仅当对任意数据$D={x_1,...,x_m}$，核矩阵$K$总是半正定的。
$$
\mathbf{K}=\left[\begin{array}{ccccc}
{\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{m}\right)} \\
{\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\
{\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{m}\right)} \\
{\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\
{\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{1}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{j}\right)} & {\cdots} & {\kappa\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{m}\right)}
\end{array}\right]
$$
![](kernel.png)
&emsp;&emsp;若$κ_1,κ_2$为核函数则：
- 对任意正数$γ_1,γ_2$其线性组合$γ_1κ_1+γ_2κ_2$也是核函数；
- 两者的直积$κ_1⊗κ_2(x,z)=κ_1(x,z)κ_2(x,z)$也是核函数；
- 对任意函数$g(x)$,$κ(x,z)=g(x)κ_1(x,z)g(z)$也是核函数。

## 1.5 软间隔

## 1.6

## 参考
[对偶问题](https://blog.csdn.net/fkyyly/article/details/86488582)
