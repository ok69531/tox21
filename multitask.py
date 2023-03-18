import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

import os
import numpy as np
import pandas as pd

from mgssl.model import GNN_graphpred
from module.train import train, eval
from module.argument import get_parser
from module.common import (
    set_seed,
    MoleculeDataset, 
    scaffold_split,
    random_scaffold_split
)


parser = get_parser()
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


criterion = nn.BCEWithLogitsLoss(reduction = "none")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

if args.task_name == 'all':
    num_task = 12
else:
    num_task = 1

os.remove('dataset/tox21/processed/pre_filter.pt')
os.remove('dataset/tox21/processed/pre_transform.pt')
os.remove('dataset/tox21/processed/geometric_data_processed.pt')

dataset = MoleculeDataset('dataset/' + args.dataset, dataset = args.dataset, task_name = args.task_name)
smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header = None)[0].tolist()

# vals, tests = [], []

# for seed in args.seed:
    
#     print('===== seed ' + str(seed))
    
best_val, final_test = 0, 0

set_seed(args.seed)

if args.split == 'scaffold':
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print('scaffold')
elif args.split == 'random_scaffold':
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.splitseed)
    print('random scaffold')
else:
    raise ValueError('Invalid split option')

args.num_workers = 0
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
val_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

model = GNN_graphpred(args.num_layer, args.emb_dim, num_task, 
                        JK = args.JK, 
                        drop_ratio = args.dropout_ratio, 
                        graph_pooling = args.graph_pooling, 
                        gnn_type = args.gnn_type)
model.from_pretrained(args.input_model_file)
model.to(device)

model_param_group = []
model_param_group.append({'params': model.gnn.parameters()})
if args.graph_pooling == 'attention':
    model_param_group.append({'params': model.pool.parameters(), 'lr': args.lr * args.lr_scale})
model_param_group.append({'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale})
optimizer = optim.Adam(model_param_group, lr = args.lr, weight_decay = args.decay)


for epoch in range(1, args.epochs + 1):
    print('=== epoch ' + str(epoch))
    train(model, device, train_loader, criterion, optimizer)
    
    if args.eval_train:
        train_loss, train_auc, train_auc_list = eval(model, device, train_loader, criterion)
    else:
        print('omit the training accuracy computation')
        train_auc = 0
    
    val_loss, val_auc, val_auc_list = eval(model, device, val_loader, criterion)
    test_loss, test_auc, test_auc_list = eval(model, device, test_loader, criterion)
    
    if val_auc > best_val:
        best_epoch = epoch
        best_val = val_auc
        best_val_auc_list = val_auc_list
        
        final_test = test_auc
        final_test_auc_lit = test_auc_list
        
    print("train_loss: %f val_loss: %f test_loss: %f" %(train_loss, val_loss, test_loss),
            "\ntrain_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))

    # vals.append(best_val)
    # tests.append(final_test)

# print(f"Average val accuracy: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
# print(f"Average test accuracy: {np.mean(tests):.3f} ± {np.std(tests):.3f}")


'''
    multitask
    seed = 0
    best epoch = 68
    best validation auc = 0.8018527888724805
    test auc = 0.8129643508135072
    
    per task auc when multitask training
    [0.7840297359651371,
    0.8611093502377178,
    0.874906015037594,
    0.7431912681912681,
    0.7394947252617724,
    0.8229166666666666,
    0.8148807170159483,
    0.8088326990474961,
    0.7942508854295364,
    0.8218251485767908,
    0.8819394663480684,
    0.8081955319840892]
'''