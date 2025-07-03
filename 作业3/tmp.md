![image-20250518004230099](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250518004230099.png)

```python
beta = 1.0
# learning_rate = 0.0003
learning_rate = 0.0003

# 训练参数
num_epochs = 200
batch_size = 64
# batch_data = 128
# batch_size = 256

white_threshold = 0.7
white_region_loss_coef = 1.0
black_region_loss_coef = 0.3

# 模型实例化
cvae = CVAE(3, 64, 32, MAX_DIGIT)
# cvae = CVAE(3, 64, 64, MAX_DIGIT)
```

```
Epoch [200/200], Step [220/235], Loss: 130.0999755859375
```

![image-20250517195343458](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250517195343458.png)

```python
beta = 1.0
# learning_rate = 0.0003
learning_rate = 0.0003

# 训练参数
num_epochs = 500 # 试一下10000
batch_size = 64
# batch_data = 128
# batch_size = 256

white_threshold = 0.5
white_region_loss_coef = 1.0
black_region_loss_coef = 0.3

# 模型实例化
cvae = CVAE(3, 64, 32, MAX_DIGIT)
# cvae = CVAE(3, 64, 64, MAX_DIGIT)
```

```
sum Loss:Epoch [500/500], Step [220/235], Loss: 71.30024719238281
```



![image-20250517171225859](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250517171225859.png)

```python
beta = 1.0
learning_rate = 0.00005

# 训练参数
num_epochs = 300
batch_size = 64

white_threshold = 0.5
white_region_loss_coef = 1
black_region_loss_coef = 0.1

cvae = CVAE(3, 64, 16, MAX_DIGIT)
```

```
0.013710260391235352
```

![image-20250512164035431](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250512164035431.png)

```python
beta = 1
learning_rate = 0.0001

# 训练参数
num_epochs = 150
batch_size = 64

white_threshold = 0.5
white_region_loss_coef = 1.5
black_region_loss_coef = 0.1
```

```
0.017893627285957336
```

![image-20250511170431052](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250511170431052.png)

AE

![image-20250511171507142](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250511171507142.png)

VAE

![image-20250511171543389](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250511171543389.png)

```python
beta = 1
learning_rate = 0.0003

# 训练参数
num_epochs = 300
batch_size = 128

white_threshold = 0.5
white_region_loss_coef = 1.0
black_region_loss_coef = 0.3
```

```
0.0273393914103508
```

![image-20250511163257243](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250511163257243.png)

```python
# 超参数调整：减小beta，加大学习率
beta = 2
learning_rate = 0.0001

# 训练参数
num_epochs = 300
batch_size = 128

white_threshold = 0.5
white_region_loss_coef = 1.0
black_region_loss_coef = 0.3
```

```
0.029147768393158913
```

![image-20250511164411015](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250511164411015.png)

```
下面结合上述可视化结果，从理论层面分析AE与VAE在潜在空间分布上的差异：

AE 的潜在空间

通常会在TSNE图中形成紧密但非规则的簇，对应不同数字类别。由于AE只优化重构误差，它能自由地将相似样本映射到相近位置，但缺少先验分布约束，导致潜在表示往往呈碎片化、不连续的簇，簇与簇之间可能出现较大“空洞”或不均匀密度分布。

VAE 的潜在空间

由于在训练中额外引入了KL散度项，VAE被强制让潜在分布尽量贴近标准正态分布。TSNE图上通常表现为更加连续、均匀的点云，类别簇之间无明显的“空洞”，整体呈环状或高斯球面的投影形态。此外，不同类别点之间会有一定程度的重叠，这正是因为VAE鼓励潜变量可连续采样，利于生成器从潜空间任意位置采样并生成合理样本。

理论解释

重构 vs 正则的权衡：AE只关注重构，潜空间可以任意分布以最小化重构误差；VAE在ELBO（证据下界）中增加了KL正则项，使潜空间分布贴近先验，从而牺牲一部分重构质量以换取更好的生成连贯性和潜空间结构化。

生成能力：VAE的连续潜空间能够从任何潜在向量生成样本，适合“插值”与多样化生成；AE潜空间缺乏结构化先验，若从未见过的潜在区域采样，往往生成不合理或失真图像。

平滑性：VAE潜在空间的平滑性更高，有助于下游任务（如分类、聚类）获得一致性表示；AE表示在类别边界附近可能出现断层，不利于这些任务。

通过TSNE可视化，直观地看到AE的簇状、碎片化分布和VAE的连续、规则分布，印证了两者在潜空间学习目标上的根本差异。 
```

