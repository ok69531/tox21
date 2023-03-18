import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, criterion):
    model.eval()
    
    y_true = []
    y_scores = []
    loss = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y.view(pred.shape)
        y_true.append(y)
        y_scores.append(pred)
        
        is_valid = y**2 > 0
        loss_mat = criterion(pred, (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        loss_ = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.append(loss_)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    # loss = torch.cat(loss, dim = 0).cpu().numpy()

    auc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            auc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(auc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(auc_list))/y_true.shape[1]))

    return sum(loss)/len(loss), sum(auc_list)/len(auc_list), auc_list 
