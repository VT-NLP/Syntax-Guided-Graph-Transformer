import argparse
model_names = ['bert-base-uncased', 'bert-large-uncased']
data_folder = './data/'


def define_arguments(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='matres', choices=['tb_dense', 'matres'])
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=7.530100210192558e-05)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6174883141474811)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--pre_training', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--event', action='store_true')
    parser.add_argument('--batch_sz', type=int, default=20, help="batch size")
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--max_iteration', type=int, default=500)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    return args

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()

    return args

