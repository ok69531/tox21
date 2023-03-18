import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--task_name', type=str, default='all', help='all, NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53')
    parser.add_argument('--input_model_file', type=str, default = 'mgssl/pre_trained.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--splitseed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="random_scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1, help='number of workers for dataset loading')

    return parser