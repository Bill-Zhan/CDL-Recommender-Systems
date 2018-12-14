import numpy as np
from os.path import exists
import re

def read_user(f_in='cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R

def read_rating(path, data_name, num_users, num_items, a, b, test_fold, random_seed):

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    R = np.zeros((num_users,num_items))  #R matrix: user-item
    mask_R = np.zeros((num_users, num_items))
    C = np.ones((num_users, num_items)) * b

    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    #--- read user item
    if data_name == 'doc':
        num_train_ratings = 0
        num_test_ratings = 0

        train_file_name = 'cf-train-1-users.dat'
        test_file_name = 'cf-train-1-users.dat'

        ''' load train fold '''
        #--- train R
        train_R = read_user(path+"/"+train_file_name)
        train_mask_R = train_R
        C = np.where(train_R==1,a,C)

        #--- train set statistics
        num_train_ratings = np.sum(train_R)
        train_record_ix = np.argwhere(train_R==1)
        user_train_set = set(train_record_ix[:,0])
        item_train_set = set(train_record_ix[:,1])
        # with open(path + train_file_name) as f1:
        #     lines = f1.readlines()
        #     for line in lines:
        #         user, item, voting = line.split("\t")
        #         user = int(user)
        #         item = int(item)
        #         voting = int(voting)
        #         if voting == -1:
        #             voting = 0

        #         ''' Total '''
        #         R[user, item] = voting
        #         mask_R[user, item] = 1

        #         ''' Train '''
        #         train_R[user, item] = int(voting)
        #         train_mask_R[user, item] = 1
        #         C[user, item] = a

        #         user_train_set.add(user)
        #         item_train_set.add(item)
        #         num_train_ratings = num_train_ratings + 1

        ''' load test fold '''
        #--- test_R
        test_R = read_user(path+"/"+test_file_name)
        test_mask_R = train_R
        #--- test set statistics
        num_test_ratings = np.sum(test_R)
        test_record_ix = np.argwhere(test_R==1)
        user_test_set = set(test_record_ix[:,0])
        item_test_set = set(test_record_ix[:,1])

        # with open(path + test_file_name) as f2:
        #     lines = f2.readlines()
        #     for line in lines:
        #         user, item, voting = line.split("\t")
        #         user = int(user)
        #         item = int(item)
        #         voting = int(voting)
        #         if voting == -1:
        #             voting = 0

        #         ''' Total '''
        #         R[user, item] = voting
        #         mask_R[user, item] = 1

        #         ''' Test '''
        #         test_R[user, item] = int(voting)
        #         test_mask_R[user, item] = 1

        #         user_test_set.add(user)
        #         item_test_set.add(item)

        #         num_test_ratings = num_test_ratings + 1
        ''' total '''
        R = np.where((train_R==1)|(test_R==1),1,0)
        mask_R = R
    assert num_train_ratings == np.sum(train_mask_R)
    assert num_test_ratings == np.sum(test_mask_R)
    # assert num_total_ratings == num_train_ratings + num_test_ratings

    return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set

# def read_trust(path,data_name, num_users):
#     if (data_name == 'politic_new') or (data_name == 'politic_old'):
#         T = np.load(path + "user_user_matrix.npy")
#     else:
#         raise NotImplementedError("ERROR")
#     return T

def read_mult(f_in='mult.dat',D=8000):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines),D))
    for i,line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        for strr in strs:
            segs = strr.split(':')
            X[i,int(segs[0])] = float(segs[1])
    arr_max = np.amax(X,axis=1)
    X = (X.T/arr_max).T
    return X

def get_mult(mult_file):
    X = read_mult(mult_file,8000).astype(np.float32)
    return X

def read_bill_term(path,data_name,num_items,num_voca):
    mult_file = path + 'mult.dat'
    X_dw = get_mult(mult_file)
    return X_dw
