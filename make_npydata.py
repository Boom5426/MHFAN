import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''
cell_root = './data/CoNIC'

a , b = 0, 0
try:
    shanghaiAtrain_path = cell_root + '/train/images/'
    shanghaiAtest_path = cell_root + '/test/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            a += 1
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/CoNIC_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            b += 1
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/CoNIC_test.npy', test_list)
    print('\n', "train:", a, "test",b)
    print("generate image list successfully")
except:
    print("The dataset path is wrong. Please check you path.")
