import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Load_data import load_data_conv
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from sklearn.manifold import TSNE
from time import time
import Nmetrics
from ICCM import MvDEC
import matplotlib.pyplot as plt
import warnings
import random
import tensorflow as tf
import builtins
import functools
import yaml
from box import Box
import string
import cupy as cp
import json
import pickle
builtins.print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 error，隐藏 warning/info
# tf.get_logger().setLevel('ERROR')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def MSE(y_true, y_pred):
    encoder_pre = y_pred[:,0,:]
    degrade_pre = y_pred[:,1,:]
    # loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(encoder_pre, degrade_pre))
    mse1 = tf.reduce_mean(tf.square(encoder_pre - tf.stop_gradient(degrade_pre)))
    # 单边 MSE: degrade_pre 有梯度，encoder_pre 停止梯度
    mse2 = tf.reduce_mean(tf.square(tf.stop_gradient(encoder_pre) - degrade_pre))
    return (mse1 + mse2) / 2

def Weighted_KLD(y_true, y_pred):
    y_true_expanded = tf.expand_dims(y_true, axis=1)  # (N, 1, D)
    view = y_pred.shape[1]
    C = tf.abs(y_pred - y_true_expanded)  # (N, K, D)
    C = tf.reduce_sum(C, axis = 2)
    C = -C
    weight_D = tf.nn.softmax(C, axis = 1) # (N, K)
    weight_D = tf.stop_gradient(weight_D)
    y_true_D = tf.repeat(y_true_expanded, view, axis = 1) #(N, K, D)
    kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    kl_values = kl(y_true_D, y_pred)
    # weighted_kl = tf.reduce_sum(1 / view  * kl_values, axis=1)  # (N,)
    weighted_kl = tf.reduce_sum(weight_D * kl_values, axis=1)  # (N,)
    final_loss = tf.reduce_mean(weighted_kl) 
    return final_loss


def _make_data_and_model(args):
    # prepare dataset
    x, y = load_data_conv(args.dataset)
    view = len(x)
    view_shapes = []
    Loss = []
    Loss_weights = []
    for v in range(view):
        view_shapes.append(x[v].shape[1:])
        Loss.append('mse')
        Loss.append(MSE)
        # Loss.append("kld")   
        Loss_weights.append(args.Idec)
        Loss_weights.append(args.dg_weight)
    Loss.append(Weighted_KLD)
    Loss_weights.append(args.lc)

    print(view_shapes)
    print(Loss)
    print(Loss_weights)
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=args.lr,
    #     decay_steps=args.maxiter
    # )

    # prepare optimizer
    optimizer = Adam(learning_rate = args.lr)
    # prepare the model
    n_clusters = len(np.unique(y))
    # n_clusters = 40   # over clustering
    print("n_clusters:" + str(n_clusters))
    # lc = 0.1

    model = MvDEC(filters=[32, 64, 128, 10],num_samples=y.shape[0],  n_clusters=n_clusters, view_shape=view_shapes, args = args)
    # tf.config.run_functions_eagerly(True)   # 让所有 tf.function 都用 eager
    model.compile(optimizer=optimizer, loss=Loss, loss_weights=Loss_weights)
    return x, y, model


def train(args):
    # get data and mode
    x, y, model = _make_data_and_model(args)

    model.model.summary()
    # pretraining
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.train_ae is False and os.path.exists(args.pretrain_dir):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrain_dir)
        # model.load_weights(args.pretrain_dir)
    else:  # train
        optimizer = Adam(lr=args.pre_lr)
        model.pretrain(x, y, optimizer=optimizer, epochs=args.pretrain_epochs,
                            batch_size=args.pre_batch_size, save_dir=args.save_dir, verbose=args.verbose)
        args.pretrain_dir = args.save_dir + '/ae_weights.h5'
    t1 = time()
    print("Time for pretraining: %ds" % (t1 - t0))

    # clustering
    # DEMVC, IDEC, DEC
    # y_pred, y_mean_pred = model.fit(arg=args, x=x, y=y, maxiter=args.maxiter,
    #                                            batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
    #                                            save_dir=args.save_dir)
    # ICCM
    indices = model.new_fit(arg=args, x=x, y=y, maxiter=args.maxiter,
                                    batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                    save_dir=args.save_dir, args = args)
    # if y is not None:
    #     for view in range(len(x)):
    #         print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
    #                 (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view])))
    #     print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
    #               (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.ari(y, y_mean_pred)))

    # t2 = time()
    # print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    # print('='*60)
    return indices


def test(args):

    x, y, model = _make_data_and_model(args)
    model.model.summary()
    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)
    y_pred, y_pred_confuse = model.predict_label(x=x)
    confuse_matrices = []
    if y is not None:
        for view in range(len(x)):
            print('Final: acc=%.4f, nmi=%.4f, ari=%.4f, pur=%.4f' %
                    (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view]), Nmetrics.pur(y, y_pred[view])))
            confuse_matrices.append(Nmetrics.get_confusion_matrix(y, y_pred[view]))
        print('Final: acc=%.4f, nmi=%.4f, ari=%.4f, pur=%.4f' %
                  (Nmetrics.acc(y, y_pred_confuse), Nmetrics.nmi(y, y_pred_confuse), Nmetrics.ari(y, y_pred_confuse), Nmetrics.pur(y, y_pred_confuse)))
        confuse_matrices.append(Nmetrics.get_confusion_matrix(y, y_pred_confuse))
    with open(args.save_dir + '/confuse_matrices.pkl', 'wb') as f:
        pickle.dump(confuse_matrices, f)
    print('End testing:', '-' * 60)

def substitute_variables(value, variables):
	if isinstance(value, str):  # 只替换字符串
		return string.Template(value).safe_substitute(variables)
	return value  # 其他类型（int, float, bool）不变



import argparse

data = 'Caltech-5V'
# data = "ALOI"
parser = argparse.ArgumentParser(description='main')
parser.add_argument('-d', '--dataset', default=data,
                    help="which dataset")
parser.add_argument('--config-path', default='./config', type=str)
temp_args = parser.parse_args()
config_dict = {}
config_file = os.path.join(temp_args.config_path, temp_args.dataset + ".yaml")
with open(config_file, 'r') as f:
    config_dict = yaml.safe_load(f)
config_dict = {k: substitute_variables(v, config_dict) for k, v in config_dict.items()}
config_dict = Box(config_dict)

args = argparse.Namespace(**config_dict)

if __name__ == "__main__":


    set_seed(args.seed)
    acc = []
    nmi = []
    ari = []
    pur = []
    for i in range(len(args.dg_weights)):
        for j in range(len(args.lcs)):
            args.dg_weight = args.dg_weights[i]
            args.lc = args.lcs[j]
            print(f'lc is {args.lc}, dg_weights is {args.dg_weight}')
            print('+' * 30, ' Parameters ', '+' * 30)
            print(args)
            print('+' * 75)
            # testing
            if args.testing:
                test(args)
            else:
                temp_indices = train(args)
                # print('-------------testing------------------')
                # test(args)
