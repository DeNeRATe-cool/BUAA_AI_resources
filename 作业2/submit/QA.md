# A2-QA

### Q1

1. **前向传播矩阵形式**

	**输入层到隐藏层**：

	输入 $x = [x _ 1, x _ 2]$，隐藏层有 3 个节点，故权重矩阵为 $W _ 1$ 为 $2 \times 3$，偏置项 $b _ 1$ 为 $1 \times 3$，因此，隐藏层输入为 $Z _ 1 = x \cdot W _ 1 + b _ 1$，其中得到 $Z _ 1$ 为 $1 \times 3$ 的向量矩阵，通过激活函数输出为 $A _ 1 = \sigma(Z _ 1)$

	**隐藏层到输出层**：

	权重矩阵 $W _ 2$ 为 $3 \times 1$，偏置矩阵 $b _ 2$ 为 $1 \times 1$，隐藏层输出为 $Z_2 = A_1 \cdot W _ 2 + b _ 2$，结果为 $A _ 2 = \sigma (Z _ 2)$

2. **交叉熵损失函数如何衡量预测概率分布与真实分布的差异？写出二分类问题的交叉熵公式**

	交叉熵损失函数可以衡量预测值和真实值之间的差异
	$$
	L = -(y \log \hat y + (1 - y) \log (1 - \hat y) )
	$$
	其中 $y$ 是真实标签，$\hat y$ 是预测概率

	$t = 1$ 时，损失为 $-\log y$，如果 $y$ 接近 1，损失小，接近 0，损失大

	$t = 0$ 时，损失为 $-\log (1 - y)$，如果 $y$ 接近 0，损失小，接近 1，损失大

### Q2

1. **在前馈神经网络中，所有的参数能否被初始化为0？如果不能，能否全部初始化为其他相同的值？原因是什么？**

	不可以全部初始化为 0，也不能全部初始化为相同的值

	如果参数初始化为相同值，则同一层的所有神经元在正向传播时会生成相同的结果，反向传播的时候也是会得到相同的梯度更新，会致使神经元之间无法学习差异化的特征，降低模型的表达能力

2. **计算权重并说明梯度下降的更新方向**

	![image-20250417193005277](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250417193005277.png)

### Q3

1. **证明**：

	![image-20250417193015365](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250417193015365.png)

2. **证明**：

  ![image-20250417193020500](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250417193020500.png)

  **画图**：

  $\sigma(x)$

  ![image-20250417170650412](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250417170650412.png)

  $\sigma ^ {'}(x)$

  ![image-20250417170845661](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250417170845661.png)

3. **证明**

	第 $i$ 层输出结果
	$$
	线性变换\quad u^{(i)} = W^{(i)} y ^ {(i - 1)} + b^{(i)}\\
	激活函数\quad y ^ {(i)} = f ^ {(i)} (u ^ {(i)})
	$$
	我们需要求的是该层参数 $\theta ^ {(i)}$ 对于损失函数 $L$ 的梯度 $\frac{\part L}{\part \theta ^ {(i)}}$

	首先，梯度可以拆分为
	$$
	\frac{\part L}{\part \theta ^ {(i)}} = \frac{\part L}{\part y ^ {(i)}} \frac{\part y ^ {(i)}}{\part \theta ^ {(i)}}
	$$
	前者记录为“损失对本层输出”的灵敏度

	后者记录为“本层参数对本层输出”的雅可比矩阵

	进而进行链式法则递推
	$$
	\frac{\part L}{\part y ^ {(i)}} = \frac{\part L}{\part y ^ {(D)}} \prod _ {k = i + 1} ^ {D} \frac{\part y ^ {(k)}}{\part y ^ {(k - 1)}}
	$$
	对于第 $i$ 层
	$$
	\frac{\partial y^{(i)}}{\partial W^{(i)}} = \frac{\partial y^{(i)}}{\partial u^{(i)}}\;\frac{\partial u^{(i)}}{\partial W^{(i)}} = \bigl[f^{(i)\,'}(u^{(i)})\bigr]\;\bigl[y^{(i-1)}\bigr]^{\!\top} \\\frac{\partial y^{(i)}}{\partial b^{(i)}} = f^{(i)\,'}(u^{(i)})
	$$
	合并后 $\theta^{(i)}=[W^{(i)},\,b^{(i)}]$，即得到了 $\frac{\part y ^ {(i)}}{\part \theta ^ {(i)}}$

	可以发现对于每个 $k$，$\frac{\part y ^ {(k)}}{\part y ^ {(k - 1)}}$ 中都包含对于 $\sigma$ 的导数

	由于
	$$
	0 < \sigma ^ {'} (u ^ {(k)}) \leq \frac{1}{4}
	$$
	因此当网络很深时
	$$
	\prod _ {k = i} ^ {D} \sigma ^ {'} (u ^ {(k)}) \leq (\frac{1}{4}) ^ {L - i + 1}
	$$
	随着 $D$ 深度的增大，其值呈指数级衰减，导致梯度消失

	