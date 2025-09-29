import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

with open('./results/Caltech-5V/confuse_matrices.pkl', 'rb') as f:
  cons = pickle.load(f)
for view in range(len(cons)):
	con = cons[view]
	con_norm = con.astype('float') / con.sum(axis=1)[:, np.newaxis]  # 归一化
	con_norm = np.around(con_norm, decimals=2)


	# 画图（混淆矩阵）
	plt.figure(figsize=(8, 6))
	ax = sns.heatmap(con_norm, vmin=0, vmax=1, annot=False, cmap='Blues')

	# 设置坐标轴标签
	# plt.xlabel('pred', fontsize=20, color='k)
	# plt.ylabel('true')

	# 设置坐标轴刻度字体大小
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)

	# 设置坐标轴的具体内容
	ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])   # x轴写标记的位置，等价于ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
	ax.set_xticklabels(['0','1','2','3','4','5','6','7'])   # x轴写标记的内容
	ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
	ax.set_yticklabels(['0','1','2','3','4','5','6','7'])
	# 设置colorbar的刻度字体大小
	cax = plt.gcf().axes[-1]
	cax.tick_params(labelsize=20)
	if view < len(cons) - 1:
		file_name = f'./results/Caltech-5V/matrix_view_{view + 1}.pdf'
	else:
		file_name = f'./results/Caltech-5V/matrix_confuse.pdf'
	# 保存为png或pdf文件
	plt.savefig(file_name, dpi=600)