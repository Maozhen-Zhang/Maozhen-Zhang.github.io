---
layout: distill
title: An Efficient Framework for Clustered Federated Learning 阅读
description: 个性化模型学习,通过经验损失进行聚类
date: 2023-03-24
tags: Papers
categories: Communication-Read

authors:
  - name: Avishek Ghosh
    url: "https://sites.google.com/view/avishekghosh/home"
    affiliations:
      name: Centre for Machine Intelligence and Data Science, Indian Institute of Technology
  - name: Jichan Chung
    url: "https://scholar.google.com/citations?user=pXQfWTkAAAAJ&hl=en"
    affiliations:
      name: Dept of EECS, UC Berkeley
  - name: Dong Yin
    url: "https://scholar.google.com/citations?user=YtM8P88AAAAJ&hl=en"
    affiliations:
      name: DeepMind


bibliography: 2023-03-22-IFCA.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Self Comprehension
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Related work
  - name: Problem formulation
  - name: Algorithm
  - name: Theoretical guarantees
  - name: Experiments
  - name: Broader Impact
# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---
## Self Comprehension

作者将聚类任务放在客户端上，而不是服务器上(一次聚类)

> 有点降低了服务器的计算，缺点时客户端需要上传自己所属的集群

## Introduction

提出迭代联邦聚类算法(IFCA)，交替估计用户的集群身份并通过梯度下降算法优化用户集群的模型参数。

> 在具有平方损失的线性模型中分析算法的收敛速度，分析通用的强凸函数和平滑损失函数

## Related work

**从非 i.i.d. 中学习单一全局模型：**[49、34、21、37、24、30]。[38、36、8]

```markdown
MOCHA，考虑多任务学习设置，形成一个确定性优化问题，其中用户的相关矩阵是正则化项，
```

**通过全局模型微调进行个性化：**

```markdown
数据作为元学习问题 [4, 15, 8]。在此设置中，目标是首先获得一个全局模型，然后每个设备使用其本地数据微调模型。该公式的基本假设是不同用户之间的数据分布相似，全局模型可以作为一个很好的初始化
```

\[36\]\[10\]考虑了FL的制定，两项工作均采用集中式的聚类算法，如K-means

> 其中服务器必须识别所有用户的集群身份，导致中心计算成本过高，这些算法可能不适用于深度神经网络等大型模型或具有大量模型的应用程序。

**IFCA**

潜在变量问题，作者提出的公式可以被视为分布式设置中具有潜在变量的统计估计问题

> 而潜在变量是统计学和非凸优化中的经典话题，如高斯混合模型GMM[44、20]，线性回归混合模型[7、43、48]和相位检索[9,29]

> 解决这些问题的两种流行方法是：期望最大化(EM)和交替最小化（AM）

近年来EM和AM在集中式环境中的融合方面的理解取得了一些进展[31, 5, 47, 1, 42]

> 如从一个合适的点开始，它们的收敛速度会很快，有时它们会有超线性的收敛速度[44, 11]

## Problem formulation

集群划分：$$m$$台机器划分为$$k$$个不相交的集群，假设不知道每台机器所属的集群

集群内数据点：假设每个客户端$$i\in S^*_j$$包含数据集$$\mathcal D_j$$的$$n$$个$$i.i.d$$数据点$$\{z^{i,1},z^{i,2},...,z^{i,n}\}$$，

> 每个数据点$$z^{i,j}$$由特征和相应的标签组成$$z^{i,\ell}=(x^{i,\ell},y^{i,\ell})$$

让$$f(\theta;z)：\Theta\rightarrow\mathbb R$$表示与数据点$$z$$相关的损失函数，$$\Theta\subseteq \mathbb R^d$$是参数空间【作者选择$$\Theta= \mathbb R^d$$】

作者的目标是最小化对所有$$j\in[k]$$的总体损失$$F^j(\theta):=\mathbb E_{z\sim\mathcal D_j}[f(\theta;z)]$$

> 拓展：这里的$$E_{z\sim\mathcal D_j}[f(\theta;z)]$$表示函数$$f$$关于样本空间$$\mathcal D_j$$的期望，指的是最大似然估计(Maximum Likelihood Estimation)，在数据集合$$\mathcal D_j$$已知的前提下， 对于函数$$f$$的参数$$\theta$$的极大似然估计值。
>
> 换句话说，希望可以得到最小化期望时的$$\theta$$值

试图找到$$\{\hat \theta\}^k_{j=1}$$使得$$\theta^*_j=\text {argmin}_{\theta\in\Theta}F^j(\theta),j\in[k]$$

原文：”因为只有有限数据，所以利用了经验损失函数“

> 不太理解这句话，如果不是因为这个还有其他的可选么

$$Z\subseteq\{z^{i,1},z^{i,2},...,z^{i,n}\}$$是第$$i$$台机器上数据点的子集

> 或者说：第$$i$$台机器上的数据点$$Z\subseteq\{z^{i,1},z^{i,2},...,z^{i,n}\}$$是子集?

定义经验损失为$$F_i(\theta;Z)=\frac{1}{\vert Z\vert}\sum_{z\in Z}f(\theta;z)$$，用$$F_i(\theta)$$来表示第$$i$$-th worker的经验损失

## Algorithm

算法主要思想：估计集群身份时交替使用最小化损失函数，讨论了IFCA的两种变体，即梯度平均和模型平均

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230322165524883.png" alt="image-20230322165524883" style="zoom:50%;" />

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230322165302302.png" alt="image-20230322165302302" style="zoom: 25%;" />

算法从$$k$$个模型参数参数开始，$$\theta^{(0)}_j,j\in[k]$$。在第$$t$$-th迭代中，IFCA的中心服务器随机选择客户端的子集，$$M_t\in[m]$$并广播当前模型参数$$\{\theta_j^{(t)}\}_{j=1}^k$$给$$M_t$$中的用户（称$$M_t$$为参与了设备的集合）

**通过参数评估**：每个客户端都配备了局部经验损失函数$$F(\cdot)$$使用收到的参数评估$$F_i$$，第$$i$$-th个客户端($$i\in M_t$$)==通过寻找最低损失的模型参数来估计其所在的集群标识==，如$$\hat j=\text{argmin}_{j\in[k]}F_i(\theta^{(t)}_j)$$

**通过梯度平均**：如果选择梯度平均选项，客户端讲计算参数$$\theta^{(t)}_{\hat j}$$的局部经验损失$$F_i$$的随机梯度，并将其集群身份估计和梯度法送回中心机器。

> $$m$$是所有客户端、$$M_t$$是随机挑选的部分客户端(子集)、$$k$$是集群个数、$$\theta_j^{(0)}$$是某一个集群的全局模型参数
>
> 【理解】：使用损失的多少来作为分类标准（如普通k-means的欧氏距离）

中心服务器接收到所有客户端的梯度和集群身份估计，收集集群身份估计相同的客户端，对各相应集群中的模型参数进行梯度下降更新，如果选择平均选项(类似于联邦平均算法[27])，每个参与的设备需要运行$$\tau$$次本地随机梯度下降更新，并且发送新模型及其集群对服务器的身份估计。然后中心服务器对集群身份估计相同的客户端新模型进行平均

### 具体实施（多任务学习权重共享）

集群结构可能是模糊的，这意味着尽管来自不同集群的数据分布不同，但模型应该利用所有用户数据的一些共同属性。

> 基于此，使用"多任务学习权重共享技术[3]"

具体而言，训练神经网络模型时，在所有集群之间共享前几层的权重，这样可以通过可获得的数据学习到良好的表示。

然后仅在最后（或最后几层上）运行IFCA算法以解决不同集群中相应的不同的分布。

使用算法1中与$$\theta_j^{(t)}$$相符的子集运行IFCA，对剩余的运行联邦平均或vanilla梯度平均

这样的好处是中心服务器不需要发送$$k$$个模型给所有机器，只需要发送所有权重的一个子集，它具有$$k$$个不同版本，以及一个共享的副本层

当集群身份的估计在几次并行迭代中没有改变认为稳定

## Theoretical guarantees

总共有$$T$$次并行迭代，将每个客户端上的$$n$$个数据点划分为$$2T$$次不相交的子集，每个子集有$$n^\prime=\frac{n}{2T}$$个数据

> 如第$$i$$-个客户端上，使用子集$$\widehat Z^{(0)}_i,...,\widehat Z^{(T-1)}_i$$聚类，使用子集$$Z^{(0)}_i,...,Z^{(T-1)}_i,$$梯度下降

使用$$\hat Z_i^{(t)}$$来聚类，用$$Z_i^{(t)}$$来计算梯度

算法的每次迭代使用新的样本数据，使用心得数据点集获得聚类估计并计算梯度

> 目的是消除聚类估计和梯度计算之间的相互依赖，并确保在每次迭代中使用新的独立同分布

即第$$j$$个簇的参数向量的更新规则为：

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230325232437953.png" alt="image-20230325232437953" style="zoom:50%;" />

$$S^{(t)}_j$$表示第$$t$$次迭代中集群身份估计为$$j$$的客户端集合，后续作者讨论了两种模型下的收敛性保证:

- 5.1节分析了具有高斯特征和平方损失的线性模型下的算法
- 5.2节分析了强凸损失函数的常规设置下的算法

在第 5.2 节中讨论了广泛研究的混合线性回归问题 [46、47] 的分布式公式。

------



### 线性模型的平方损失

假设第$$j$$-簇聚类中客户端的数据产生的方式是：$$i\in S^*_j$$，第$$i$$个客户端的特征反应满足
$$
y^{i,\ell}=\langle x^{i,\ell},\theta^*_j\rangle+\epsilon^{i,\ell}
$$
其中$$x^{i,\ell}~\sim\mathcal N(0,I_d)$$，独立于$$x^{i,\ell}的$$加性噪声(additive noise)$$\epsilon^{i,\ell}\sim\mathcal N(0,\sigma^2)$$，如我们所见，该模型是分布式线性回归模型的混合，在上述设置下， 参数$$\{\theta^*_j\}^k_{j=1}$$是总损失函数$$F^j(\cdot)$$的最小值

> 加性噪声$$\epsilon^{i,\ell}$$与$$x^{i,\ell}的$$无关

1. 将$$p_i:=\vert S^*_j\vert/m$$作为第$$j$$个聚簇中客户端数量占总数的比例，

2. 并让$$p:=\min\{p_1,p_2,...,p_k\}$$，

3. 同样定义最小分离$$\Delta$$，$$\Delta:=\min_{j\not=j^\prime\Vert \theta^*_j-\theta^*_{j^\prime}\Vert}$$，$$\rho:=\frac{\Delta^2}{\sigma^2}$$作为信噪比

在确定收敛结果前，陈述一些假设，回想$$n^\prime$$表示每步操作中每个客户端的数据数量

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326095722576.png" alt="image-20230326095722576" style="zoom:50%;" />

在假设中，作者假设初始化足够接近$$\theta^*_j$$，并指出到这是混合模型[1, 45]的收敛分析中的标准假设，因为这是混合模型问题的非凸优化，在假设2中，我们对$$n^\prime,m,p,d$$做出了温和的假设。条件$$pmn^\prime\gtrsim d$$的简单假设，我们每次迭代时对于每个集群全部数据的大小至少和参数空间维度一样大，条件$$$$\Delta \gtrsim \frac{\sigma}{p} \sqrt{\frac{d}{m n^{\prime}}}+\exp \left(-c\left(\frac{\rho}{\rho+1}\right)^2 n^{\prime}\right)$$$$保证迭代接近$$\theta^*_j$$

作者对算法进行了单步分析，假设在某个迭代中，获得了接近真实$$\theta^*_j$$的参数向量$$\theta_j$$，并且表明$$\theta_j$$以指数速率收敛于$$\theta^*_j$$并带有一个误差底线

> 作者假设初始化足够接近$$\theta^*_j$$，这是混合模型[1,45]收敛分析的标准假设，
>
> 作者在假设2中，对$$n^\prime,m,p,d$$做出了宽松的假设，$$pmn^\prime\gtrsim d$$，$$d$$是参数的维度

**Theorem 1.**考虑线性模型并假设Assumptions 1和2成立。假设在某次迭代中得到的参数向量$$\theta_j$$满足$$\Vert\theta_j-\theta^*_j\Vert\le\frac{1}{4}\Delta$$

让$$\theta^+_j$$表示这次迭代后的向量，存在通用常数$$c_1,c_2,c_3,c_4>0$$，这样当我们选择步长$$\gamma=c_1/p$$的概率至少为$$1-1/\text{poly}(m)$$，对所有的$$j\in[k]$$，我们有：

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326113410093.png" alt="image-20230326113410093" style="zoom:50%;" />

附录中证明了定力1，简要总结思路：

- 使用初始化条件，作者表明集合$$\{S_j\}^k_{j=1}$$与$$\{S^*_j\}^k_{j=1}$$有显著的重叠
- 在重叠集合中，认为因为线性回归的基本属性，梯度的步骤存在了收缩和误差。
- 然后作者限制了错误分类客户端的梯度范数并将它们添加到误差层
- 作者通过结合正确分类和错误分类的客户端的贡献来证明
- 迭代应用Therem1并在一下推论中获得最终解$$\widehat \theta_j$$的精度

**Corollary 1.**  考虑到线性模型并假设Assumptions 1和2成立。通过选择$$\gamma=c_1/p$$的概率至少为$$$$1-\frac{\log (\Delta / 4 \varepsilon)}{\operatorname{poly}(m)}$$$$,在并行迭代$$T=\log\frac{\Delta}{4\epsilon}$$次后，我们对于所有$$j\in[k]$$，有$$\Vert\theta_j-\theta^*_j\Vert\le\frac{1}{4}\Delta$$，其中$$$$\varepsilon=c_5 \frac{\sigma}{p} \sqrt{\frac{d}{m n^{\prime}}}+c_6 \exp \left(-c_4\left(\frac{\rho}{\rho+1}\right)^2 n^{\prime}\right)$$$$

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326161424049.png" alt="image-20230326161424049" style="zoom:50%;" />

检验最终的正确性：

由于每个客户端的数据点$$n=2n^\prime T=2n^\prime\log(\Delta/4\epsilon)$$，我们知道对于最小的集群，总共有$$2pmn^\prime\log(\Delta/4\epsilon)$$个数据点，根据线性回归的minimax rate[41]，我们能知道即使知道真实的聚类身份，我们也无法获得比$$\mathcal O(\sigma\sqrt{\frac{d}{pmn^\prime\log(\Delta/4\epsilon)}})$$更优的错误率。我们统计正确率$$\epsilon$$与此错误率相比，可以看到第一项$$\frac{\sigma}{p}\sqrt\frac{d}{mn^\prime}$$在$$\epsilon$$中与minimax rate相当，只是存在一个对数因子和一个关于数据维度$$p$$的依赖关系。同时，该算法的第二项误差随着样本量$$n^\prime$$的增加而指数级衰减，因此最终的统计误差接近最优的水平

------

### Strongly convex loss functions

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326150844392.png" alt="image-20230326150844392" style="zoom:50%;" />

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326150908592.png" alt="image-20230326150908592" style="zoom:50%;" />

Assumption 3:假设总的损失函数$$F^j(\theta)$$是强凸函数和光滑的，不会对单个损失函数$$f(\theta,z)$$做出convexity或smoothness的假设，相反对$$f(\theta;z)$$和$$\nabla f(\theta;z)$$下的分布假设：

Assumption 4:对于每个$$\theta$$和$$j\in[k]$$，当$$z$$是$$\mathcal D_j$$的随机采样，$$\eta^2$$是$$f(\theta;z)$$的方差上界，即$$$$\mathbb{E}_{z \sim \mathcal{D}_j}\left[\left(f(\theta ; z)-F^j(\theta)\right)^2\right] \leq \eta^2$$$$

Assumption 5:对于每个$$\theta$$和$$j\in[k]$$，当$$z$$是$$\mathcal D_j$$的随机采样，$$v^2$$是$$\nabla f(\theta;z)$$的方差上界，即$$$$\mathbb{E}_{z \sim \mathcal{D}_j}\left[\Vert\nabla f(\theta ; z)-\nabla F^j(\theta)\Vert^2_2\right] \leq v^2$$$$



梯度的有界方差在分析SGD[6]中非常常见。

本文中作者使用损失函数来确定聚类身份，因此还需要对$$f(\theta;z)$$的进行概率假设。作者表明方差的有界性约束是相对较弱的假设约束，除了上述的假设，仍然使用5.1节中的一些定义，如：

- 最小间隔$$\Delta$$，$$\Delta:=\min_{j\not=j^\prime\Vert \theta^*_j-\theta^*_{j^\prime}\Vert}$$

- 将$$p_i:=\vert S^*_j\vert/m$$，

​		作为第$$j$$个聚簇中客户端数量占总数的比例，

- $$p:=\min\{p_1,p_2,...,p_k\}$$

对初始化$$n^\prime,p,\Delta$$做出如下假设：

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326152538630.png" alt="image-20230326152538630" style="zoom:50%;" />

为了简单起见，$$\widetilde {\mathcal O}$$ 符号省略了任何不依赖于$$m$$和$$n^\prime$$的对数因子和数量，正如我们所见的，我们需要假设良好的初始化，因为混合模型的性质和我们对$$n^\prime,p,\Delta$$相对宽松的假设，特别是$$\Delta$$假设确保了迭代失踪保持靠近在$$\theta^*_j$$的$$\ell_2$$距离的球面上。



<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326154556848.png" alt="image-20230326154556848" style="zoom:50%;" />

假设Assumptions 3-6成立，选择步长$$\gamma=1/L$$，然后在概率至少为$$1-\delta$$的情况下，并行迭代$$T=\frac{8L}{p\lambda\log(\frac{\Delta}{2\epsilon})}$$次，对所有$$j\in[k],\Vert\widehat \theta_j-\theta^*_j\Vert\le\epsilon$$，其中

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326155009530.png" alt="image-20230326155009530" style="zoom:50%;" />

附录B中证明了定理2，与5.1节类似，为了证明这个结果，首先需要证明每次迭代的收缩：

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230326155132692.png" alt="image-20230326155132692" style="zoom:50%;" />

然后获得收敛速度，为了更好的解释结果，作者关注于对m和n的依赖性，并将其他量视为常熟，然后由于$$n=2n^\prime T$$，我们知道$$n$$和$$n^\prime$$在对数银子上具有相同的比例。因此可以得到最终的计算误差为$$$$\epsilon=\widetilde{\mathcal{O}}\left(\frac{1}{\sqrt{m n}}+\frac{1}{n}\right)$$$$。如5.1节所述，即使知道集群身份，$$\frac{1}{\sqrt{mn}}$$也是最佳速率，因此作者的统计率在接近$$n\gtrsim m$$的情况下接近最优。于线性模型中的统计率相比$$$$\widetilde{\mathcal{O}}\left(\frac{1}{\sqrt{m n}}+\exp (-n)\right)$$$$，作者注意到主要区别在于第二项。线性模型和强凸情况下的附加项分别是$$\exp(-n)$$和$$\frac{1}{n}$$，作者注意到这是因为由于不同的统计假设：线性模型中，假设高斯噪声，而强凸情况下，只假设有界方差。

------

## 实验

不会再每次迭代时重新采样新数据点，此外，还可以放宽初始化要求

> More specifically，对于线性模型，我们观察到随机初始化和几次重新启动足以确保算法1的收敛

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230323102018587.png" alt="image-20230323102018587" style="zoom:33%;" />

> 不同成功的概率:
>
> - (a)、(b)是分离尺度$$R$$和加性噪声$$\sigma$$
> - (c)、(d)是客户端数量$$m$$和每个客户端采样大小$$n$$
>
> (a)和(b)中，随着$$R$$的增加成功的概率增加，即，基本反映出真实参数向量的之间的距离程度
>
> (c)和(d)中，随着$$mn$$的增加成功的概率在提升，即每个客户端的数据更多/客户端的数量更多，成功的概率也会提升

### 生成数据

首先在具有平方损失的线性模型上使用梯度平均(选项I)评估算法,

首先生成$$\theta^*_j\sim\text{Bernoulli(0.5)}的值，并且将它们的$$$$\ell_2$$范数作为$$R$$，这确保了$$\theta^*_j$$之间的间距与$$R$$成正比期望

> $$\theta^*_1,...,\theta^*_j$$

每次实验中，首先生成参数向量$$\theta^*_j$$并固定他们，根据独立的伯努利分布对应的随机初始化$$\theta^{(0)}_j$$

运行算法1，300次迭代，步长不变。对于$$k=2$$和$$k=4$$，分别在$$\{0.01,0.1,1\}$$和$$\{0.5,1.0,2\}$$中选择步长。为了确定是否成功学习了模型，我们回到上述步长并定义距离的度量：$$\text{dist}=\frac{1}{k}\sum^k_{j=1}\Vert \hat \theta_j-\theta^*_j\Vert$$，其中$$\{\hat \theta_j\}^k_{j=1}$$是从算法1中获得的参数估计，如果对于$$\theta^*_j$$的固定集合，在10个随机初始化参数$$\theta_j^{(0)}$$中，至少在一个场景中获得$$\text {dist}\le0.6\sigma$$，则实验称为成功

在图2(a-b)中，针对分离参数$$R$$绘制了40次实验的经验成功概率。将问题参数设置为

1. (a),  $$k=2并且(m,n,d)=(100,100,1000)$$
2. (b),  $$k=4并且(m,n,d)=(400,100,1000)$$

正如所看到的，当$$R$$变大时，即参数之间的距离会变大，问题变得更容易解决，成功概率更高。这验证了作者的理论结果，更高的信噪比产生更小的误差层。

在图2(c-d)中，作者描述了对$$m,n$$的依赖性，将$$R$$和$$d$$固定为

1. (c),  $$(R,d)=(0.1,1000)$$
2. (d),  $$(R,d)=(0.5,1000)$$

观察到，当增加$$m$$或$$n$$的数量时，成功的概率会提高

### MNIST和CIFAR的旋转

基于MNIST[19]和CIFAR-10[18]数据集创建了聚类FL数据集。为模拟不同机器上的数据从不同分布生成的环境，使用==旋转扩充数据集==

并创建旋转MNIST[25]和旋转CIFAR数据集

> 对MNIST应用0、90、180、270度旋转来扩充数据集，从而产生$$k=4$$个簇，对于给定的$$m和n$$满足$$mn=60000k$$，将图像随机划分为$$m$$个客户端，每个客户端上有$$n$$个具有相同旋转的图像。同样用相同的方式拆分测试数据集$$m_{test}=10000k/n$$个客户端
>
> 对于CIFAR数据集，与MNIST相似操作，拆分主要区别在于创建了$$k=2$$个具有0度和190度旋转的簇，

作者注意到，通过操纵MNIST和CIFAR-10等标准数据集来创建不同的任务已在持续学习研究社区中得到广泛采用[12、16、25]。对于集群FL，使用旋转创建数据集有助于我们模拟具有清晰集群结构的联邦学习设置。

对于MNIST实验，使用具有ReLU激活的券链接神经网络，单个隐藏层大小为200，CIFAR实验，使用2个卷积层和2个全连接层组成的卷机神经网络模型，图像通过标准数据增强（翻转、随机裁剪）进行预处理

作者将IFCA算法与两种基线算法进行比较，即全局模型和局部模型方案。对于IFCA，我们使用模型平均(算法1中的选项II)/

> 对于MNIST实验，我们使用完整的客户端参与(对所有$$t$$，$$M_t=[m]$$)。对于算法1的本地更新，我们选择$$\tau=10$$并且补偿$$\gamma=0.1$$。对于CIFAR实验，选择$$\vert M_t\vert=0.1m$$，并且应用下降的补偿0.99，还为LocalUpdate的过程设置$$\tau=5$$，batch size 50
>
> 遵循之前的工作[28]（fedAvg）
>
> 1. 在全局模型方案中，该算法尝试学习一个单一的全局模型，该模型可以从所有分布中进行预测。该算法不考虑聚类身份，因此算法1中的模型平均的操作变成$$\theta^{(t+1)}=\sum_{i\in M_t}\hat \theta/\vert M_t\vert$$，即对所有参与机器参数进行平均。
> 2. 在局部模型的方案中每个解ID那种的模型仅对局部可用数据进行梯度下降，不进行模型平均

对于IFCA和全局方案，通过以下方式进行推理：

> 每台测试客户端，我们对所有学习模型(IFCA的$$k$$个模型和一个全局模型模型)进行推理，并从产生最小损失的模型计算准确率。
>
> 为了测试局部模型基线，在相同分布的测试数据测试准确率(如那些旋转的数据)

作者展示了客户端中所有模型的平均准确度，对于所有算法，使用5个不同的随机种子进行实验并报告平均值和标准偏差

实验结果如表1所示，可以观察到作者的算法比两个基线性能更好，当运行IFCA算法时，作者观察到可以逐渐找到工作机器底层集群的标识，并且在找到正确的集群后，使用具有相同分布的数据训练和测试每个模型，从而获得更好的准确性。全局基线模型的性能比作者提出算法的性能更差。

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230324104723562.png" alt="image-20230324104723562" style="zoom:50%;" />

### Fedrated Emnist

作者在Federated Emnist(FEMNIST)[2]上提供了额外的实验结果，使用4.1节中提到的权重共享技术。

使用具有两个卷机层+一个最大池化层+两个全连接层的神经网络，共享所有层的权重，除了IFCA训练的最后一层。将簇的数量$$k$$视为超参数，并使用不同的$$k$$进行实验，对比了具有IFCA的全局模型和局部模型的方法，并且还与一次性集中聚类算法进行了比较。测试精度如表2其中计算了5次独立运行的平均值和标准偏差。如所看到的，IFCA比全局模型和局部模型方法显出明显的优势，

> IFCA和One-shot算法的结果是相似的，但是如第2节中强调的，IFCA不运行集中式的聚类程序，因此降低了服务器的计算成本。
>
> 最后，作者观察到IFC对于簇数$$k$$的选择是稳健的。$$k=2$$和$$k=3$$的算法结果类似，并且注意到当$$k>3$$时，IFCA自动识别出3个簇，其余簇为空这表明IFCA在聚类结构不明确切簇类数量未知的现实问题中的适用性

<img src="https://mz-pico-1311932519.cos.ap-nanjing.myqcloud.com/image/image-20230324104741100.png" alt="image-20230324104741100" style="zoom:50%;" />

## Broader Impact

作者提出它们的框架将更好地保护联邦学习系统中用户的隐私，同时仍提供个性化预测，

> 不需要用户将自己的任何个人数据发送到中央服务器，用户仍然可以使用服务器的计算能力学习个性化模型
>
> 一个潜在的风险是作者的算法仍然需要用户将集群身份发送到中央服务器。
