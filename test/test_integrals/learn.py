#! /usr/bin/env python
from momo import osio, endl, flush
from __pyosshell__ import get_dirs
import numpy as np

def extract_data(folder, N_train, N_test, regex):
    def extract_data_sub(subfolder, transform_y = lambda y: np.log10(y)):
        base = osio.cwd()
        osio.cd(subfolder)
        X = np.load('xnkl.array.npy')
        y = float(open('jeff2_h.txt', 'r').readline())
        y = transform_y(y)
        osio.cd(base)
        return X, y
    def sort_subfolders(folders):
        return sorted(folders, key = lambda f: int(f.split('_')[1])*len(folders)+int(f.split('_')[2]))

    base = osio.cwd() 
    osio << "Base =" << base << endl
    osio.cd(folder)

    # GET SUBFOLDERS
    subfolders = sort_subfolders(get_dirs('./', regex))
   
    # LIMIT RANGE 
    N = N_train + N_test
    assert N < len(subfolders)
    subfolders = subfolders[:N]
    
    # FIND DESCRIPTOR DIMENSIONS
    dim_x = extract_data_sub(subfolders[0])[0].shape[0]
    osio << "Descriptor dimension =" << dim_x << endl

    # EXTRACT FEATURES & TARGETS
    X_list = np.zeros((N, dim_x))
    y_list = np.zeros((N))
    idx = -1
    for subfolder in subfolders:
        idx += 1
        osio << osio.back << subfolder << flush
        X, y = extract_data_sub(subfolder)
        X_list[idx] = X
        y_list[idx] = y
    osio << endl

    # SPLIT ONTO TRAINING & TEST SET
    X_train = np.zeros((N_train, dim_x))
    y_train = np.zeros((N_train))
    X_test = np.zeros((N_test, dim_x))
    y_test = np.zeros((N_test))


    X_train[0] = X_list[0]
    y_train[0] = y_list[0]
    count_train = 1
    count_test = 0
    for idx in range(1,N):
        if float(count_train)/(count_train+count_test) < float(N_train)/(N):
            X_train[count_train] = X_list[idx]
            y_train[count_train] = y_list[idx]
            count_train += 1
        else:
            X_test[count_test] = X_list[idx]
            y_test[count_test] = y_list[idx]
            count_test += 1
    
    assert count_train == N_train
    assert count_test == N_test

    osio.cd(base)
    return X_train, y_train, X_test, y_test





# TODO Take log of tintegrals
# TODO Raise kernel to power of xi






N_train = 3000
N_test = 500
xi = 3.
lambda_reg = 1e-2

X_train, y_train, X_test, y_test = extract_data('frame0', N_train, N_test, 'pair_')

def compute_kernel(X1, X2, xi):
    osio << "Compute kernel ..." << endl
    K = np.dot(X1, X2.T) # = np.inner(X1,X2)
    osio << "Normalize ..." << endl
    norm_1 = np.sqrt(np.sum(X1*X1,1))
    norm_2 = np.sqrt(np.sum(X2*X2,1))
    norm_12 = np.outer(norm_1, norm_2)
    K = K/norm_12
    osio << "Sharpen ..." << endl
    K = K**xi
    return K

K = compute_kernel(X_train, X_train, xi)
np.savetxt('kernel.array.txt', K)

osio << "Regularize ..." << endl
K_reg = K + lambda_reg*np.identity(K.shape[0])

osio << "Invert ..." << endl
K_reg_inv = np.linalg.inv(K_reg)
np.savetxt('kernel_inv.array.txt', K_reg_inv)

osio << "Compute weights ..." << endl
w = K_reg_inv.dot(y_train)
np.savetxt('weights.array.txt', w)

osio << "Train ..." << endl
y_train_calc = np.dot(K, w)
rms_error_train = (np.sum((y_train_calc-y_train)**2)/y_train.shape[0])**0.5
print rms_error_train
np.savetxt('y_train_calc.array.txt', y_train_calc)
np.savetxt('y_train_ref.array.txt', y_train)



# Optimize xi and lambda
# Neural network to single out important components?
# PCA
# Try different descriptor

osio << "Test ..." << endl
K_test = compute_kernel(X_test, X_train, xi)
y_test_calc = np.dot(K_test, w)
rms_error_test = (np.sum((y_test_calc-y_test)**2)/y_test.shape[0])**0.5
print rms_error_test
np.savetxt('y_test_calc.array.txt', y_test_calc)
np.savetxt('y_test_ref.array.txt', y_test)



"""
osio << "Compute kernel matrix ..." << endl
K = np.dot(X_train, X_train.T)

osio << "Normalize ..." << endl
norm_vec_sqrt = np.sqrt(np.sum(X_train*X_train,1))
norm_mat_sqrt = np.outer(norm_vec_sqrt, norm_vec_sqrt)
K = K/norm_mat_sqrt

osio << "Sharpen ..." << endl
K = K**xi
np.savetxt('kernel.array.txt', K)

osio << "Regularize ..." << endl
K_reg = K + lambda_reg*np.identity(K.shape[0])

osio << "Invert kernel ..." << endl
K_reg_inv = np.linalg.inv(K_reg)
np.save('kernel_inv.array.npy', K_reg_inv)
np.savetxt('kernel_inv.array.txt', K_reg_inv)

osio << "Compute weights ..." << endl
w = K_reg_inv.dot(y_train)
np.save('weights.array.npy', w)
np.savetxt('weights.array.txt', w)
"""


