import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from deepfm import deepfm


def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            pred += list(y_hat.numpy())
            target += list(y.numpy())
    auc = roc_auc_score(target, pred)
    return auc


if __name__=="__main__":

    batch_size = 1024
    lr = 0.00005
    wd = 0.00001
    epoches = 1

    seed = 1024
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features
    df = pd.read_csv('/mnt/ssd/dataset/kaggle/train.txt', names=col_names, sep='\t')
    df = df.sample(frac=0.1)
    df.to_csv("/mnt/ssd/dataset/kaggle/train_sample.txt", index=True)
    
    feature_names = dense_features + sparse_features

    df[sparse_features] = df[sparse_features].fillna('-1', )
    df[dense_features] = df[dense_features].fillna(0, )
    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    feat_size1 = {feat: 1 for feat in dense_features}
    feat_size2 = {feat: len(df[feat].unique()) for feat in sparse_features}
    print(feat_size2)
    feat_sizes = {}
    feat_sizes.update(feat_size1)
    feat_sizes.update(feat_size2)

    # print(df.head(5))
    # print(feat_sizes)

    train, test =train_test_split(df, test_size=0.2, random_state=2021)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    device = 'cuda:0'

    model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
                   dnn_hidden_units=[1000, 500, 250], embedding_size=16,
                   l2_reg_linear=1e-3, device=device)

    train_label = pd.DataFrame(train['label'])
    train_data = train.drop(columns=['label'])
    #print(train.head(5))
    train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)), torch.from_numpy(np.array(train_label)))
    train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

    test_label = pd.DataFrame(test['label'])
    test_data = test.drop(columns=['label'])
    test_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_data)),
                                                       torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size)

    loss_func = nn.BCELoss(reduction='mean')
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epoches):
        total_loss_epoch = 0.0
        total_tmp = 0

        model.train()
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).float()

            y_hat, indexes = model(x)

            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_tmp += 1
            
            diff = {}
            for name,p in model.state_dict().items():
                if "embedding" in name:
                    col = name.split('.')[1]
                    print(col)
                    updated_emb={}
                    for idx in indexes[col]:
                        idx = int(idx.item())
                        updated_emb[idx] = p[idx].detach().data.cpu()
                    diff[name] = updated_emb
                    print("differential size,", len(updated_emb)," param size, ",p.size()[0])
              
            torch.save(diff, "diff.pt")
            torch.save(model, "model.pt")
            break

        # auc = get_auc(test_loader, model) 
        # print('epoch/epoches: {}/{}, train loss: {:.3f}, test auc: {:.3f}'.format(epoch, epoches, total_loss_epoch / total_tmp, auc))
        