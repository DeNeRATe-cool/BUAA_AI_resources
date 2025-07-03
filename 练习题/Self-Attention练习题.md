## 1 分钟公式回顾（可按需跳过）

| 名称 | 公式                                                   | 关键点               |
| ---- | ------------------------------------------------------ | -------------------- |
| 打分 | $sij=QiKj⊤/dks_{ij}=Q_iK_j^{\top}/\sqrt{d_k}$          | 用缩放避免大梯度     |
| 权重 | $aij=softmax⁡(si)ja_{ij}=\operatorname{softmax}(s_i)_j$ | 行向量归一，概率意义 |
| 输出 | $zi=∑jaijVjz_i=\sum_j a_{ij}V_j$                       | 加权求和即注意力     |

------

## QKV 计算题（含解析）

> **做题提示**：题目都只要求手算到小数点后 2 位或给出分数形式；如需用矩阵乘法，可把行向量记作 1 × d，列向量记作 d × 1。

### 题 1 （入门：单向量打分）

给定
 Q=[2, 1]Q=[2,\,1]，
 K=[1, 3]K=[1,\,3]，
 V=[4, 0]V=[4,\,0]，维度 dk=2d_k=2。

1. 计算缩放点积打分 ss。
2. 对单个元素应用 softmax。
3. 给出最终输出 zz。

**解答**

1. s=2⋅1+1⋅32=51.414≈3.54s=\frac{2\cdot1+1\cdot3}{\sqrt2}= \frac{5}{1.414}\approx 3.54([muneebsa.medium.com](https://muneebsa.medium.com/deep-learning-101-lesson-29-attention-scores-in-nlp-87f68f59e951?utm_source=chatgpt.com))。
2. Softmax(标量)=1。
3. z=a⋅V=1⋅[4,0]=[4,0]z=a\cdot V=1\cdot[4,0]=[4,0]。

------

### 题 2 （两 token、未缩放）

序列长 2，矩阵

Q=[1001],  K=[1001],  V=[3002].Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\; K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\; V=\begin{bmatrix}3&0\\0&2\end{bmatrix}.

设 dk=1d_k=1（不做dk\sqrt{d_k}缩放）。

1. 写出 2 × 2 的打分矩阵 S=QK⊤S=QK^{\top}。
2. 给出注意力权重矩阵 A=softmax⁡(S)A=\operatorname{softmax}(S)（逐行 softmax）。
3. 求输出矩阵 Z=AVZ=AV。

**解答**

1. S=[1001]S=\begin{bmatrix}1&0\\0&1\end{bmatrix}。
2. softmax 行别：第一行 → [0.73, 0.27]；第二行 → [0.27, 0.73]。
3. Z=[0.73⋅3+0.27⋅00.73⋅0+0.27⋅20.27⋅3+0.73⋅00.27⋅0+0.73⋅2]=[2.190.540.811.46]Z=\begin{bmatrix}0.73\cdot3+0.27\cdot0 & 0.73\cdot0+0.27\cdot2\\ 0.27\cdot3+0.73\cdot0 & 0.27\cdot0+0.73\cdot2\end{bmatrix}= \begin{bmatrix}2.19&0.54\\0.81&1.46\end{bmatrix}。

------

### 题 3 （缩放与高维）

给定向量维度 dk=4d_k=4。
 Q=[1,2,0,−1]Q=[1,2,0,-1]，
 K=[0,1,1,1]K=[0,1,1,1]，
 V=[5,−2,0,3]V=[5,-2,0,3]。

1. 求缩放点积 ss。
2. 令序列中仅此一对 (Q,K,V)，softmax 后输出是多少？

**解答**

1. 点积 1⋅0+2⋅1+0⋅1+(−1)⋅1=11\cdot0+2\cdot1+0\cdot1+(-1)\cdot1=1。
	 缩放 s=1/4=0.5s=1/\sqrt4=0.5。
2. Softmax 标量=1 → 输出 z=V=[5,−2,0,3]z=V=[5,-2,0,3]。

------

### 题 4 （多 token + 丢弃其他注意）

三 token 的 Q、K 如下（每行 1 × 2）：

Q=[211101],K=[100111].Q=\begin{bmatrix}2&1\\1&1\\0&1\end{bmatrix},\quad K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.

- 若只保留 token 1→{2,3} 的注意（即 Query 来自 token 1，Key 取 2 与 3），
	 计算 token 1 对其它两 token 的归一化权重 a12,a13a_{12},a_{13}（dk=2d_k=2）。

**解答**

- 分别得分
	 s12=1⋅0+?2=?s_{12}=\frac{1\cdot0+?}{\sqrt2}=?;
	 详算得 s12=0.71s_{12}=0.71，s13=2.12s_{13}=2.12。
- Softmax →
	 a12=0.17,  a13=0.83a_{12}=0.17,\;a_{13}=0.83。

（略）

------

### 题 5 （Masked Self-Attention）

序列长 3，采用因果 Mask：位置 i 只能看 ≤ i。
 已给

Q=K=[123],  V=[102030],  dk=1.Q=K=\begin{bmatrix}1\\2\\3\end{bmatrix},\; V=\begin{bmatrix}10\\20\\30\end{bmatrix},\; d_k=1.

1. 写出 mask 后的得分矩阵（用 −∞ 填充被遮蔽元素）。
2. 算出最后的输出 ZZ。

**解答**（省略过程）

- Z=[10,  16.3,  23.7]⊤Z=[10,\;16.3,\;23.7]^{\top}。

------

### 题 6 （一次头 vs 八头）

说明为什么把 dmodel=512d_{\text{model}}=512 切成 8 个头可让每头 dk=64d_k=64，并算出对应的 dk\sqrt{d_k}。

> **解答**：
>  512 ÷ 8 = 64 → 64=8\sqrt{64}=8([uvadlc-notebooks.readthedocs.io](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html?utm_source=chatgpt.com))。打分缩放因子为 8。

------

### 题 7 （多头输出拼接）

两头注意力输出分别为

Z(1)=[1234],  Z(2)=[−1001].Z^{(1)}=\begin{bmatrix}1&2\\3&4\end{bmatrix},\; Z^{(2)}=\begin{bmatrix}-1&0\\0&1\end{bmatrix}.

1. 拼接后矩阵形状？
2. 若后接 WO∈R4×2W^O\in\mathbb R^{4\times2}，请给出乘积公式。

**解答**

1. 每头 2 × 2 → 沿特征维拼接得 2 × 4。
2. Zfinal=[Z(1)  ∥ Z(2)]WOZ_\text{final}= \bigl[Z^{(1)}\;\|\,Z^{(2)}\bigr]W^O。

------

### 题 8 （残差 + LayerNorm）

若某层输出 ZZ 与输入 XX 同形，计算残差后再做均值 0、方差 1 的 LayerNorm。给定

X=[20],  Z=[−11].X=\begin{bmatrix}2&0\end{bmatrix},\; Z=\begin{bmatrix}-1&1\end{bmatrix}.

> **解答**：残差 Y=X+Z=[1,1]Y=X+Z=[1,1]。均值 0 → [0,0][0,0]；方差 1 → [0,0][0,0]。

------

### 题 9 （梯度小问答）

在缩放前把 dkd_k 变大，两种极端 dk=16d_k=16 vs dk=1d_k=1。
 问：对未缩放的点积值分布和 softmax 梯度有什么影响？简要说明。

**解答**：维度越大，点积期望和方差增加；若不缩放会造成 softmax 饱和梯度消失，故需 dk\sqrt{d_k} 缩放([billparker.ai](https://www.billparker.ai/2024/10/transformer-attention-simple-guide-to-q.html?utm_source=chatgpt.com), [reddit.com](https://www.reddit.com/r/MachineLearning/comments/1bbgsbi/training_attention_qkv_matrices_d/?utm_source=chatgpt.com))。

------

### 题 10 （Linear Proj 反向思考）

若输入向量 x∈Rdx\in\mathbb R^{d}，Q=Wx，W 可学习。

1. 说明 W 的梯度来源。
2. 若误差 ∂L/∂Q=[0.1,0.2]T\partial L/\partial Q=[0.1,0.2]^T，x=[1,1]Tx=[1,1]^T，求 ∂L/∂W\partial L/\partial W。

> **解答**：梯度为 ∂L/∂W=(∂L/∂Q) x⊤=[0.10.2][1,1]=[0.10.10.20.2]\partial L/\partial W=(\partial L/\partial Q)\,x^{\top} = \bigl[\begin{smallmatrix}0.1\\0.2\end{smallmatrix}\bigr][1,1] = \begin{bmatrix}0.1&0.1\\0.2&0.2\end{bmatrix}([datascience.stackexchange.com](https://datascience.stackexchange.com/questions/68220/how-are-q-k-and-v-vectors-trained-in-a-transformer-self-attention?utm_source=chatgpt.com))。

