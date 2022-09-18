# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict 
import spacy
import networkx as nx
from spacy.gold import align



"""
" Dependency mappings
"""
deps = {
    "acl":0,
    "acomp":1,
    "advcl":2,
    "advmod":3,
    "agent":4,
    "amod":5,
    "appos":6,
    "attr":7,
    "auxpass":8,
    "case":9,
    "cc":10,
    "ccomp":11,
    "compound":12,
    "conj":13,
    "cop":14,
    "csubj":15,
    "dative":16,
    "dep":17,
    "det":18,
    "dobj":19,
    "expl":20,
    "intj":21,
    "mark":22,
    "list":23,
    "mark":24,
    "meta":25,
    "neg":26,
    "nn":27,
    "nounmod":28,
    "npmod":29,
    "nsubj":30,
    "nsubjpass":31,
    "nummod":32,
    "oprd":33,
    "obj":34,
    "obl":35,
    "parataxis":36,
    'pcomp':37,
    'pobj':38,
    'poss':39,
    'preconj':40,
    'prep':41,
    'prt':42,
    'punct':43,
    'quantmod':44,
    'relcl':45,
    'root':46,
    'xcomp':47,
    'aux':48,
    'nmod':49,
    'npadvmod':50,
    'cross-sentence':51,
    'predet':52,
    'csubjpass':53,
    '': 54
}
def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else float(numr) / float(denr)
import itertools


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
    
    

    
import pickle
from torch.utils import data

"""
" Event dataset from Rujun's paper for TB-dense
"""
class EventDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, data_split):
        'Initialization'
        # load data
        with open(data_dir + data_split + '.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
            self.data = list(self.data.values())
        handle.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
#         print(self.data[idx].keys())
#         print(self.data[idx]['left_event'].id)
        sample = self.data[idx]
        doc_id = sample['doc_id']
        context = sample['doc_dictionary']
        rels = sample['rel_type']
        left_event = sample['left_event']
        right_event = sample['right_event']
        events = sample["event_labels"]

        return doc_id, context,  rels, left_event, right_event, events
#         return sample, doc_id
class Event():
    id: str
    type: str
    text: str
    tense: str
    polarity: str
    span: (int, int)

"""
" Process data to indices
"""
def data_read(datatype, bert_model)
    train_data_read = EventDataset("../data/", datatype)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    lens = []
    doc_dict = dict()
    rels_all = dict()
    sents_dict = dict()
    events_dict = dict()
    event_ids_dict = dict()
    for idx, i in enumerate(train_data_read): 
        sent = []
        events = []
        indices = []
        sent_id = 0
        start_idx = 0
        write_flag = False
        if i[0] not in doc_dict:
            doc_dict[i[0]] = dict()
            rels_all[i[0]] = dict()
            sents_dict[i[0]] = dict()
            events_dict[i[0]] = dict()
            event_ids_dict[i[0]] = dict()
            write_flag = True
    #     tmp = list(map(int,cur.replace("[", "").replace(")", "").split(":")))
        left = [i[3].span[0], i[3].span[1]+1][0]
        right = [i[4].span[0], i[4].span[1]+1][0]
        left_sent_idx = -1
        right_sent_idx = -1
        left_idx = -1
        right_idx = -1
        event_ids = np.array(list(i[-1].values()))
        for key, tok, eve in zip(i[1].keys(), i[1].values(), i[5].values()):
            tmp = list(map(int, key.replace("[", "").replace(")", "").split(":")))[0]
            sent.append(tok)
            events.append(eve)
            if tmp == left:
    #             print(sent)
                left_idx = start_idx
                left_sent_idx = sent_id
            elif tmp== right:
    #             print(sent)
                right_idx = start_idx 
                right_sent_idx = sent_id
            start_idx += 1
            if tok[0] == '.':
                if write_flag:
                    doc_dict[i[0]][sent_id] = sent
                    events_dict[i[0]][sent_id] = events
                sent_id += 1
                sent =[]
                events = []
                start_idx = 0

    #         start_idx += len(tokenizer.tokenize(tok[0]))

            if right_idx!= -1 and  left_idx != -1 and (left_sent_idx, right_sent_idx) not in rels_all[i[0]]:
                rels_all[i[0]][(left_sent_idx, right_sent_idx)] = list()
    #     print(left_idx, right_idx)
        if write_flag and sent_id not in doc_dict[i[0]] and len(sent) > 0:
            doc_dict[i[0]][sent_id] = sent
            events_dict[i[0]][sent_id] = events
        if right_idx == -1:
            print(right_sent_idx, right, i[1])
        if right_sent_idx > left_sent_idx:
            right_idx += len(doc_dict[i[0]][left_sent_idx])
    #     rels_all[i[0]][(left_sent_idx, right_sent_idx)].append(((left_idx, i[3].tense, i[3].polarity),
    #                                                             (right_idx, i[4].tense, i[4].polarity), i[2]))
        rels_all[i[0]][(left_sent_idx, right_sent_idx)].append((left_idx, right_idx, rel_map[i[2]]))
        if (left_sent_idx, right_sent_idx) not in sents_dict[i[0]]:
            if left_sent_idx == right_sent_idx:
                sents_dict[i[0]][(left_sent_idx, right_sent_idx)] = doc_dict[i[0]][left_sent_idx]
                event_ids_dict[i[0]][(left_sent_idx, right_sent_idx)] = events_dict[i[0]][left_sent_idx]
            else:
                if left_sent_idx not in doc_dict[i[0]] or right_sent_idx not in doc_dict[i[0]] :
                    print(doc_dict[i[0]].keys(), left_sent_idx, right_sent_idx, i[1])
                sents_dict[i[0]][(left_sent_idx, right_sent_idx)] = doc_dict[i[0]][left_sent_idx] + doc_dict[i[0]][right_sent_idx]
                event_ids_dict[i[0]][(left_sent_idx, right_sent_idx)] = events_dict[i[0]][left_sent_idx] + events_dict[i[0]][right_sent_idx]
        return sents_dict, event_ids_dict
    
"""
" Retrieve the directed and undirected grah of original sentences
" For adjacent senteces, they are connected via cross-sentence relation
"""
def dep_graph(sent, indices):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
#     nlp = spacy.load("en_trf_bertbaseuncased_lg")
    doc = nlp(sent)
    G = nx.Graph()
    diG = nx.DiGraph()
    roots = []
    for tok in doc:
        for child in tok.children:
        
            if len([(tok, child,  child.dep_) for child in tok.children]) < 1:
                continue

            if tok.dep_ == "ROOT" and tok not in roots:
                roots.append(tok)
#             diG.add_edge('{0}-{1}-{2}'.format(tok.lower_,tok.i), '{0}-{1}-{2}'.format(child.lower_), dep=child.dep_)
            diG.add_edge(indices[tok.i], indices[child.i], dep=deps[child.dep_])

            G.add_edge(indices[tok.i], indices[child.i])
            diG.edges(data=True)
        if len(roots) > 1:
            for i in range(len(roots)-1) :
                diG.add_edge(indices[roots[i].i],indices[roots[i+1].i], dep=deps["cross-sentence"])
                G.add_edge(indices[roots[i].i],indices[roots[i+1].i])
    
    return diG, G



"""
" return processed sents pos tag info
"""
def pos_tag_process(sents_dict):
    pos_set = set()
    for i in sents_dict:
        for j in sents_dict[i]:

            pos = [cur[1] for cur in sents_dict[i][j]]

            for a_pos in pos:
                pos_set.add(a_pos)

    pos_id = {k: v for v, k in enumerate(pos_set)}
    return pos_id


"""
" align spacy and bert tokenizer ids
" return tensor for first token ids for each token, relation types, pos tags, sentence tensors
"""
def gather_all_data(sents_dict, events_dict):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    rel_types = {'BEFORE': 0, 'AFTER':1, 'INCLUDES':2, 'IS_INCLUDED':3,'SIMULTANEOUS':4,'VAGUE':5 }
    from spacy.tokenizer import Tokenizer
    import pickle
    def listToString(s):  

        # initialize an empty string 
        str1 = " " 

        # return string   
        return (str1.join(s)) 
    sents = []
    masks = []
    poses = []
    final_events = []
    first_sub = []
    saved_rels = []
    graphs = []
    cur_idx = 1
    for i in sents_dict:
        for j in sents_dict[i]:
            cur_rels = []
            text = [cur[0] for cur in sents_dict[i][j]]
            pos = [pos_id[cur[1]] for cur in sents_dict[i][j]]

            text = listToString(text)
            a,b,c = convert_to_ids_tensor(text, tokenizer)
            sents.append(a)
            masks.append(b)
            poses.append(pos)
            first_sub.append(c)
            for pair in rels_all[i][j]:
                cur_rels.append((c[pair[0]], c[pair[1]], pair[2]))
            saved_rels.append(cur_rels)
            graphs.append(dep_graph(text, c))
            cur_idx+=1
    for i in sents_dict:
        for j in sents_dict[i]:
            final_events.append(events_dict[i][j])
    sents = torch.stack(sents)
    masks = torch.stack(masks)
    event_tensor = torch.stack(final_events)
    dataset = TensorDataset(sents, masks, event_tensor)
    data = [dataset, poses, saved_rels, graphs, first_sub]
    return data