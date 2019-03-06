# Local Uncertainty Sampling
## 抽样策略

对于所有数据点$(x, c)$给定抽样函数$a(x, c) \in [0, 1]$.

(1)对于每个数据点$(x_i, c_i) (i=1, \cdots ,n)$，生成二值变量$z_i \in {0,1}$，并按照$Z~服从\cal{B}(x_i,c_i)$进行抽样：
$$
\cal P_{\cal{B}(x_i, c_i)} = 
\begin{cases}
a(x_i,c_i), &z_i=1 \\\
1 - a(x_i, c_i), &z_i=0
\end{cases}
$$

(2) 保留每个$z_i=1(i \in \{1,\cdots,n\})$的样本，并进行分类。无论是二分类还是多分类。

## 算法过程

--------
Algorithm 1 LUS


1: 选择$\gamma \geq 1. $

2: 使用一个弱分类器扫描样本。得到$\hat{p}_i=(\hat{p}_{i,1}, \cdots,\hat{p}_{i,K}), \hat{p}_{i,k} 是指x_i属于类别k的概率。$（原文描述为给出一个粗糙的估计）

3： 扫描数据，并对每个样本生成随机变量 $z_i \sim \cal{B}(a(x_i,c_i):z_i=1)$, 对于每个$z_i的分布函数为$：
$$
\cal a(x_i,c_i) = 
\begin{cases}
\frac{1-\hat{q}_i}{\gamma-max(\hat{q}_i,0.5\cdot\gamma)}, &if~~\hat{p}_{i,c_i} = \hat{q}_i \geq 0.5\\\
min(1, \frac{2\hat(q)_i}{\gamma}), &\text{otherwise}\end{cases}
$$
其中，$\hat{q}_i=max(0.5,\hat{p}_{i,1}, \cdots,\hat{p}_{i,K})$

4：对于抽样样本，进行分类训练。即$z_i=1$ 的样本作为新的训练集，得到新的模型。

#Reverse Local Uncertainty Sampling
## 抽样策略

对于所有数据点$(x, c)$给定抽样函数$a(x, c) \in [0, 1]$.

(1)对于每个数据点$(x_i, c_i) (i=1, \cdots ,n)$，生成二值变量$z_i \in {0,1}$，并按照$Z~服从\cal{B}(x_i,c_i)$进行抽样：
$$
\cal P_{\cal{B}(x_i, c_i)} = 
\begin{cases}
a(x_i,c_i), &z_i=1 \\\
1 - a(x_i, c_i), &z_i=0
\end{cases}
$$

(2) 保留每个$z_i=1(i \in \{1,\cdots,n\})$的样本，并进行分类。无论是二分类还是多分类。
#多轮抽样boosting实验
--------
1: 初始化数据集$D_0, 利用\neq [T-2, T+2]的样本，初始化抽样模型M_0$

2: 使用$M_0 抽样全量样本得到D_1$, $利用M_0 对D_1的预测$（sigmoid变换前的值）作为初始化，得到$\hat{D_1}, 并用\hat{D_1}训练得到模型M_1$

3：重复操作2，得到$\hat{D_i} 和 M_i$

4：使用最终抽样基模型$M_i$, 抽样全量数据$D_{i+1}$，进行模型训练和预测。

#多轮迭代抽样实验
--------

1: 初始化数据集$D_0, 利用\neq [T-2, T+2]的样本，初始化抽样模型M_0$

2: 使用$M_0 抽样全量样本得到D_1, 并用D_1训练得到模型M_1$

3：重复操作2，得到$D_i 和 M_i$

4：使用最终抽样基模型$M_i$, 抽样全量数据$D_{i+1}$，进行模型训练和预测。

