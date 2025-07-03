# A1-QA

### Q1

![image-20250323152500568](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250323152500568.png)

### Q2

1. 因为 $Y$ 在数据集固定的时候是一个常数，即 $\lambda T ^ T T$ 是一个常数

	其在优化过程中，无法通过梯度对参数优化起到作用，因此起不到正则化的作用

2. 对于 L2 正则化，其作用是在损失函数中增加 $\omega$ 的大小项，起到对 $\omega$ 过大时的惩罚，减小过拟合的风险

	若 $\lambda < 0$ ，则效果恰好相反，$\omega$ 越大，反而可以得到更小的损失，无法达到限制 $\omega$ 的效果

### Q3

1. <img src="C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250323144413637.png" alt="image-20250323144413637" style="zoom: 67%;" />

2. 在图中，所有的正方形都在直线下方，所有的三角形都在直线上方，因此训练错误率为 **0**

3. 根据 SVM 的理论，只有**支持向量**会映像最终的分类边界，对于其他非支持向量集合中的点，移除都不会改变间隔平面的解

	因此，只有移除这些支持向量才会导致分类边界发生变化，即 G 或者 C、D

### Q4

![image-20250323152506663](C:\Users\12298\AppData\Roaming\Typora\typora-user-images\image-20250323152506663.png)