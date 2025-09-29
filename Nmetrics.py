import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def pur(y_true, y_pred):
    # y_true: 真实标签
    # y_pred: 聚类结果（或预测标签）
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_assignment(-cm)
    # 构建聚类标签到真实标签的映射
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    # 重新映射聚类标签
    y_pred_aligned = np.array([mapping[label] for label in y_pred])
    # 返回对齐后的混淆矩阵
    return confusion_matrix(y_true, y_pred_aligned)
