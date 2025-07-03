# 实验二
# 根据reg_type(l1, l2)和lambda计算正则项损失reg_loss（通常不对偏置项正则化）
# --------------------------- TODO ---------------------------------------
reg_loss = 0.0
for param in model.parameters():
    # 偏置项排除
    if 'weight' in param.name:
        if reg_type == 'l1':
            reg_loss += lambda_ * paddle.sum(paddle.abs(param))
        else:
            reg_loss += lambda_ * paddle.sum(paddle.square(param))
# ------------------------------------------------------------------------

# 实验三
# 将线性结果通过sigmoid得到概率值
# --------------------------- TODO ---------------------------------------
def forward(self, x):
    output1 = self.linear(x)
    output2 = self.sigmoid(output1)
    return output2
# ------------------------------------------------------------------------

# 使用二元交叉熵损失
# --------------------------- TODO ---------------------------------------
output = model(X_tensor)
loss = paddle.nn.functional.binary_cross_entropy(output, y_tensor)
# ------------------------------------------------------------------------

# 实验四
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_4(X_train_dtm, y_train):
    # 自行探索sklearn库里的Logistic Regression或别的模型，观察效果如何
    # 逻辑回归 0.8626943005181348
    # model = LogisticRegression(max_iter=300, solver='lbfgs')  
    # 支持向量机 0.881516587677725
    model = SVC(kernel='rbf', C=10)
    # 决策树 0.7704280155642023
    # model = DecisionTreeClassifier(max_depth=1000)
    # 随机森林 0.7967257844474762
    # model = RandomForestClassifier(n_estimators=1000)
    # 梯度提升树 0.7878787878787878
    # model = GradientBoostingClassifier(n_estimators=100)
    # K 近邻 0.1822033898305085
    # model = KNeighborsClassifier(n_neighbors=10)

    model.fit(X_train_dtm, y_train)  # 训练模型
    model_path = "sklearn.pkl"
    joblib.dump(model, model_path)  # 保存模型
    print(f"训练完成，模型已保存至：{model_path}")
    
    return model_path

def evaluate_4(model_path, X_test_dtm, y_test):
    model = joblib.load(model_path)  # 加载模型
    y_pred = model.predict(X_test_dtm)  # 预测

    return y_pred

# 训练和评估
model_path = train_4(X_train_selected, y_train)
y_pred = evaluate_4(model_path, X_test_selected, y_test)
# 评估指标输出
print("\n在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))

print("\n在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))

print("在测试集上的 f1-score：")
print(metrics.f1_score(y_test, y_pred))