import pickle
from config import *
from utils import *
from model import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


def train(args, model, args):
    
def test(test_loader, model, args):
    
    for step, data in enumerate(test_loader):


        cur_data = data[0]
        cur_mask = data[1]

        b_input_ids = cur_data.type(torch.LongTensor)
        b_input_mask = cur_mask.type(torch.LongTensor)

        if (step+1)*bs < len(rels_test_expanded):
            b_re_lab = rels_test_expanded[step*bs:(step+1)*bs]
            graphs_test = graph_data_test_all[step*bs:(step+1)*bs]
            graphs_ids = graphs_ids_test_all[step*bs:(step+1)*bs]

            graphs = [i[1] for i in graphs_test]
            di_graphs = [i[0] for i in graphs_test]
            cur_pos = one_hot_pos_test[step*bs:(step+1)*bs]

        else:

            b_re_lab = rels_test_expanded[step*bs:]
            graphs_test = graph_data_test_all[step*bs:]
            graphs_ids = graphs_ids_test_all[step*bs:]

            graphs = [i[1] for i in graphs_test]
            di_graphs = [i[0] for i in graphs_test]
            cur_pos = one_hot_pos_test[step*bs:]



        b_token_ids = torch.zeros(bs,160)
        for idx, i in enumerate(graphs_ids):
            b_token_ids[idx][i] = 1 
        b_token_ids = b_token_ids.type(torch.BoolTensor)
        if torch.cuda.is_available():
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_token_ids = b_token_ids.to(device)


        outputs_f = model(input_ids=b_input_ids, attention_mask=b_input_mask, 
                    tuples=b_re_lab, graphs=graphs,di_graphs=di_graphs,
                    token_ids=graphs_ids,pos=cur_pos, env=False)

        labels_f = torch.Tensor([t[2] for t in b_re_lab]).type(torch.LongTensor).to(device)
        y_pred = np.concatenate((y_pred, torch.argmax(outputs_f, dim=1).cpu().data.numpy()))
        y_true = np.concatenate((y_true, labels_f.cpu().data.numpy()))

        score = accuracy_score(y_true, y_pred)
        return score
    
def prepare_dataloader(args, checkpoint=None):
    
    if args.dataset == 'matres':
        pos_len = 54
    else:
        pos_len = 44
    
    """
    "Load data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGraphTransformer(device, head=args.head)
    model.cuda()

    # scheduler = get_linear_schedule_with_warmup(optimizer,1,264)

    
    enc = OneHotEncoder(handle_unknown='ignore')
    one_hot_pos_train = []
    for i in pos_train:
        cur = torch.zeros(len(i), pos_len).to(device)
        for idx, j in enumerate(i):
            cur[idx][j] = 1
        one_hot_pos_train.append(cur)
    one_hot_pos_test = []
    for i in pos_test:
        cur = torch.zeros(len(i), pos_len).to(device)
        for idx, j in enumerate(i):
            cur[idx][j] = 1
        one_hot_pos_test.append(cur)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGraphTransformer(device)
    model.cuda()
    # global_step = 0
    criterion_e = F.cross_entropy
    criterion_r = F.cross_entropy

    if args.checkpoint is not None:
#         checkpoint = torch.load('./saved_model/bert-large-sota-12layers-3e6lr-batch20.pth.tar')
        checkpoint = torch.load(args.checkpoint)
    # checkpoint = torch.load('event_e.pth')
        model.load_state_dict(checkpoint['state_dict'])    

    fr = open('./data/test_transformer_rels_space.pkl', 'rb')
    data_test = pickle.load(fr)
    test_dataset = data_test[0]
    rels_test = data_test[1]
    pos_test = data_test[2]
    with open('./data/graph_test_space.pkl', 'rb') as f:
        graph_data_test = pickle.load(f)
    # with open("pos_test.pkl", 'rb') as f:
    #     pos_test = pickle.load(f)
    test_loader = DataLoader(
            test_dataset,
            batch_size=args.bs,
            num_workers=4
        )
    fr = open('./data/train_transformer_rels_space.pkl', 'rb')
    data = pickle.load(fr)
    train_dataset = data[0]
    rels = data[1]

    with open('./data/graph_train_space.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    # with open("pos_train.pkl", 'rb') as f:
    #     pos_train = pickle.load(f)
    pos_train = data[2]
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            num_workers=4
        )

    fr = open('./data/val_transformer_rels_space.pkl', 'rb')
    data_val = pickle.load(fr)
    val_dataset = data_val[0]
    rels_val = data_val[1]
    pos_val = data_val[2]
    # with open("pos_val.pkl", 'rb') as f:
    #     pos_val = pickle.load(f)
    import pickle
    with open('./data/graph_val_space.pkl', 'rb') as f:
        graph_data_val = pickle.load(f)

    val_loader = DataLoader(
            val_dataset,
            batch_size=args.bs,
            num_workers=4
        )
def main():
    args = parse_arguments()
    bs = args.bs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.dataset == 'matres':
        
    