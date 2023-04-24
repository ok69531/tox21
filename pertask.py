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

args.task_name = 'SR-p53'

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
        final_test = test_auc
    
    print("train_loss: %f val_loss: %f test_loss: %f" %(train_loss, val_loss, test_loss),
            "\ntrain_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))

    # vals.append(best_val)
    # tests.append(final_test)

# print(f"Average val accuracy: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
# print(f"Average test accuracy: {np.mean(tests):.3f} ± {np.std(tests):.3f}")


'''
    1. NR-AR
    seed = 0
    best epoch = 1
    best validation auc = 0.8411209259823332
    test auc = 0.6648936170212766
    
    2. NR-AR-LBD
    seed = 0
    best epoch = 23
    best validation auc = 0.9084155897018419
    test auc = 0.8052614896988906
    
    3. NR-AhR
    seed = 0
    best epoch = 17
    best validation auc = 0.8570180351741907
    test auc = 0.8674624060150377
    
    4. NR-Aromatase
    seed = 0
    best epoch = 90
    best validation auc = 0.819785759294266
    test auc = 0.7836798336798337
    
    5. NR-ER
    seed = 0
    best epoch = 33
    best validation auc = 0.8017420862545146
    test auc = 0.7448578669233641
    
    6. NR-ER-LBD
    seed = 0
    best epoch = 38
    best validation auc = 0.8355738000721761
    test auc = 0.7950246710526315
    
    7. NR-PPAR-gamma
    seed = 0
    best epoch = 99
    best validation auc = 0.678562421185372
    test auc = 0.8484908395940424
    
    8. SR-ARE
    seed = 0
    best epoch = 48
    best validation auc = 0.7348906192065259
    test auc = 0.7874174890881335
    
    9. SR-ATAD5
    seed = 0
    best epoch = 18
    best validation auc = 0.8245982810164424
    test auc = 0.7582159624413145
    
    10. SR-HSE
    seed = 0
    best epoch = 21
    best validation auc = 0.7519230769230769
    test auc = 0.7943384422896467
    
    11. SR-MMP
    seed = 0
    best epoch = 98
    best validation auc = 0.8338261230418093
    test auc = 0.8844882516925527
    
    12. SR-p53
    seed = 0
    best epoch = 100
    best validation auc = 0.8119571511291257
    test auc = 0.7659420108573287
'''