---
layout: post
title: a post with twitter
date: 2020-09-28 11:12:00-0400
description: an example of a blog post with twitter
tags: formatting
categories: sample-posts external-services
<!--layout: post
title: deep learning code comprehend
date: 2023-02-28 12:10:00
description: comprehend python and deep learning come
tags: code
categories: code-comprehend Deep-Learning
-->
---

# 代码理解

## 普通网络训练时  

模型训练时：

```python
#外层epoch循环
for e in range(args.epochs):
  running_loss = 0 #记录每轮epoch的损失
  #遍历dataloader，里面的data是batch
  for i,data in enumerate(train_loader):
    #data[0]是输入数据，data[1]是标签
    inputs, labels = data[0].to(device), data[1].to(device)
    
    #一些方法的获取和初始化
    #损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    #优化器>>>实现随机梯度下降算法,lr– 学习率,momentum– 动量因子
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  
    optimizer.zero_grad()#调用backward()之前要将梯度清零
    
		#向前、向后、优化
    outputs = net(inputs)							#前向
		loss = criterion(outputs, labels)	#求损失操作：将输出和标签输入
    loss.backward()										#损失向输入测进行反向传播>>>将梯度积累到x.grad中备用
    optimizer.step()									#优化器对x值进行更新，
```



#### 细节



##### loss.backward()理解

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
loss.backward()
#即
loss = nn.CrossEntropyLoss(outputs, labels).backward()
```

将损失loss向输入侧进行反向传播，同时对于需要进行梯度计算的所有变量$x(requires_grad=True)$，计算梯度$\frac{d}{dx}loss$并将其累积到梯度$x.grad$中备用，即
$$
x.grad = x.grad+\frac{d}{dx}loss
$$

##### optimizer.step()理解

```python
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  
optimizer.zero_grad()	#调用backward()之前要将梯度清零,因为使其在每个batch中不进行累计
optimizer.step()
```

是优化器对x的值进行更新，以随机梯度下降SGD为例子：

> 学习率(learning rate, lr)来控制步幅，即$x = x - lr * x.grad$，减号是因为沿着梯度反方向调整变量以减少cost

##### optimizer.zero_grad()理解

> Pytorch 为什么每一轮batch需要设置？

根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。

还可以补充的一点是，如果不是每一个batch就清除掉原有的梯度，而是比如说两个batch再清除掉梯度，这是一种变相提高batch_size的方法，对于计算机硬件不行，但是batch_size可能需要设高的领域比较适合，比如目标检测模型的训练