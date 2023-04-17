---
layout: post
title: The Algorithmic Foundationsof Differential Privacy(三)-5.组合定理
date: 2023-04-17 13:11:00
description: The Algorithmic Foundationsof Differential Privacy第3章第五节组合定理
tags: TAFDP
categories: TAFDP(三)
published: true
---

# 组合定理

**定理3.13：**

设$\mathcal M_1:\mathbb N^{|\chi|}\rightarrow\mathcal R_1$是一个满足$\epsilon_1$-DP的算法，$\mathcal M_2:\mathbb N^{|\chi|}\rightarrow\mathcal R_2$是一个满足$\epsilon_2$-DP的算法，那么它们的组合定义为$\mathcal M_{1,2}:\mathbb N^{|\chi|}\rightarrow\mathcal R_1\times\mathcal R_2$，则$\mathcal M_{1,2}(x)=(\mathcal M_1(x),\mathcal M_2(x))$满足$\epsilon_1+\epsilon_2$-DP

> 证明：
>
> 设$x,y\in\mathbb N^{|\chi|}$满足$\Vert x - y\Vert_1\le1$，对于任意的$(r_1,r_2)\in\mathcal R_1\times\mathcal R_2$，则：
> $$
> \begin{aligned}
> \frac{\operatorname{Pr}\left[\mathcal{M}_{1,2}(x)=\left(r_1, r_2\right)\right]}{\operatorname{Pr}\left[\mathcal{M}_{1,2}(y)=\left(r_1, r_2\right)\right]} & =\frac{\operatorname{Pr}\left[\mathcal{M}_1(x)=r_1\right] \operatorname{Pr}\left[\mathcal{M}_2(x)=r_2\right]}{\operatorname{Pr}\left[\mathcal{M}_1(y)=r_1\right] \operatorname{Pr}\left[\mathcal{M}_2(y)=r_2\right]} \\
> & =\frac{\operatorname{Pr}\left[\mathcal{M}_1(x)=r_1\right]}{\operatorname{Pr}\left[\mathcal{M}_1(y)=r_1\right]} \frac{\operatorname{Pr}\left[\mathcal{M}_2(x)=r_2\right]}{\operatorname{Pr}\left[\mathcal{M}_2(y)=r_2\right]} \\
> & \leq \exp \left(\varepsilon_1\right) \exp \left(\varepsilon_2\right) \\
> & =\exp \left(\varepsilon_1+\varepsilon_2\right)
> \end{aligned}
> $$
> 也有$\frac{\operatorname{Pr}\left[\mathcal{M}_{1,2}(x)=\left(r_1, r_2\right)\right]}{\operatorname{Pr}\left[\mathcal{M}_{1,2}(y)=\left(r_1, r_2\right)\right]}\ge\exp(-(\epsilon_1+\epsilon_2))$



<font color="red">思考:</font>
值域是$\mathcal R_1\times \mathcal R_2$，所以说它们是两个算法想称，也就是输入为$x$时，落入不同区域的概率(分布可能性)，因此有：

> $\Pr[\mathcal M_1(x)=r_1]$，满足差分隐私的算法$\mathcal M_1$中输入$x$，其值落到$r_1$的概率。
>
> $\mathcal M_{1,2}(x)=(\mathcal M_1(x),\mathcal M_2(x))$  ，表示对应的两个输出，落到的总概率区间（相乘）



**推论**：设$\mathcal M_i:\mathbb N^{|\chi|}\rightarrow\mathcal R_i$满足$(\epsilon_i,0)$-DP，$i\in[k]$。如果$\mathcal M_{[k]}:\mathbb N^{|\chi|}\rightarrow \prod_{i=1}^k\mathcal  R_i$被定义为$\mathcal M_{[k]}(x)=(\mathcal M_1(x),...,\mathcal M_k(x))$，则$M_{[k]}(x)$满足$(\sum_{i=1}^k\epsilon_i,0)$-DP。



**定理3.14：**







