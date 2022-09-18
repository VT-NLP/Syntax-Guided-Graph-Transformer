###### import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import BertTokenizer
import networkx as nx
from torchvision import datasets, transforms
import math


attention_holder = []
class TemporalGraphTransformer(nn.Module):
    """
    " Event label: Event, Nonevent
    " Relation Label: Before, After Equal, Vague
    """
    def __init__(self, device, model_name, num_labels=4, num_event_classes = 2, head=8):
    #         config = BertConfig.from_pretrained('bert-base-uncased') 
        super(TemporalGraphTransformer, self).__init__()
        
        self.device = device
        self.num_labels = num_labels
        self.num_event_classes = num_event_classes
#         model_config = BertConfig.from_pretrained('bert-large-uncased', output_hidden_states=True)
#         self.bert = BertModel.from_pretrained('bert-large-uncased', config=model_config)
        model_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(model_name, config=model_config)

        self.dropout = nn.Dropout(model_config.hidden_dropout_prob*2)
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=1e-12)
        self.rel_emb  = nn.Embedding(55, model_config.hidden_size).to(device)
#         self.pos_emb  = nn.Embedding(44, model_config.hidden_size).to(device)
        """
        " Temporal graph transformer module
        " Take a graph and embeddings as input
        " Encode all nodes using graph attention
        " input should contain number of heads, input hidden dimension， output hidden dimension and graph
        """
        self.graphTransformer = TransformerModule(device)
        
        """
        " Event prediction
        " Three layers of NN for predicting each embedding as an entity
        """
        self.linear1 = nn.Linear(model_config.hidden_size, model_config.hidden_size*2)
        torch.nn.init.xavier_uniform(self.linear1.weight)
        self.linear2 = nn.Linear(model_config.hidden_size*2, model_config.hidden_size)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        self.pos_linear = nn.Linear(model_config.hidden_size*2, model_config.hidden_size)
        self.linear3 = nn.Linear(model_config.hidden_size, self.num_event_classes)
        torch.nn.init.xavier_uniform(self.linear3.weight)

        """
        " Relation prediction
        " We select events from predicted entities and we then use them to classify the relations
        " If an event is not in the relation table, we classify it as None, any pair of entities
        " with at least one non-event will have the None relation. Or we will have the gold labeled
        " relation
        """
        self.linear1_rel = nn.Linear(model_config.hidden_size*2+44*2, model_config.hidden_size)
        self.classifier = nn.Linear(model_config.hidden_size, self.num_labels)


        self.act = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.softmax_ent = nn.Softmax(dim=1)
    
    """
    " Use transformer output embeddings to calcuate the 
    """
    def forward(self, input_ids, graphs, di_graphs, tuples=None,
                token_ids = None, token_type_ids=None, 
                event_ids = None, attention_mask=None, 
                labels=None, env=True, pos=None):
        
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_state = self.LayerNorm(pooled_output[0])
#         hidden_state,_ = self.lstm(hidden_state)
        """
        " Graph transformer embeddings.
        " To avoid influences across different event pair, each pair has its own copy of node representations
        " This enables us to compute in parallel via threadpool
        """
        def transformer_output(self, idx, cur_hidden_state, graph, di_graph,token_id, pair, rel_emb, input_ids):
            cur_h = self.graphTransformer(cur_hidden_state, graph, di_graph,
                                                         token_id, pair, rel_emb, input_ids)
            return cur_h
        output_state = []
        from concurrent.futures import   ThreadPoolExecutor   
        for cur_id, (cur_hidden_state, cur_tuple) in enumerate(zip(hidden_state, tuples)):
            with ThreadPoolExecutor(max_workers=10) as executor:   #To reduce the computation overhead, # of maxworkder should be less

                futures = list()
                for idx, pair in enumerate(cur_tuple):
                    futures.append(executor.submit(transformer_output, self, idx, cur_hidden_state, 
                                                   graphs[cur_id], di_graphs[cur_id],
                                                   token_ids[cur_id], pair, self.rel_emb, input_ids[cur_id]))
                for idx, (pair, cur_h) in enumerate(zip(cur_tuple, futures)):
                    cur_h = cur_h.result()
                    selected1 = torch.cat((cur_h[pair[0]], cur_pos[cur_id][token_ids[cur_id].index(pair[0])]))
                    selected2 = torch.cat((cur_h[pair[1]], cur_pos[cur_id][token_ids[cur_id].index(pair[1])]))
                    cur_output = torch.cat((selected1, selected2))
                    cur_output = self.linear1_rel(cur_output)
                    output_state.append(cur_output)
        if len(output_state) == 0:
            print(tuples)
        output_state = torch.stack(output_state)
        output_state = self.act(output_state)
        output_state = self.dropout(output_state)
#         print(output_state.size())
        logits = self.classifier(output_state)
    #         m = nn.Softmax(dim=1)

        return logits
    
    
"""
" For this customized module, the input is reduced to d_sent_len*dmodel
"""
class TransformerModule(nn.Module):
    
    def __init__(self, device, num_heads=1, dmodel=1024, demb=1024, nlayers=12, dropout=0.1):
        """
        " Step One: Initalize learning embedding of edge types
        " Step Two: Initialize learnable parameters to initalize both forward and backward representation
        " Step Three: Form Triplet and assign their paraemters
        " Step Four: Graph attention
        " Step Five: Fushion layer
        " Step Six: FFN
        " Step Seven: Eventual node representation
        """
        super().__init__()
        self.device = device
        self.hid_init_linear = nn.Linear(demb, dmodel)#Initialize the forward and backward representation in the same starting point
        torch.nn.init.xavier_uniform(self.hid_init_linear.weight)
        self.encoders = nn.ModuleList([TransformerEncoder(device) 
                                       for i in range(nlayers)])
        self.nlayers = nlayers
        self.dmodel = dmodel
        self.demb = demb
        self.sigmoid = nn.Sigmoid()
        self.fushion = nn.Linear(dmodel*2, 1)
        self.gelu = nn.GELU()
        self.res = nn.Linear(dmodel, dmodel*2)
        self.res = nn.Sequential( #sequential operation
                            nn.Linear(dmodel, dmodel*4), 
                            self.gelu, 
                            nn.Linear(dmodel*4, dmodel*2))
        self.FinalRep = nn.Linear(dmodel*2, demb, bias=False)
        
        self.LayerNorm = nn.LayerNorm(dmodel, eps=1e-12)
        

    """
    " Read the graph, due to the nature of this model one graph is processed at a time 
    " produce forward and backward embeddings,
    " produce triplet,
    " produce graph attention
    """
    def forward(self, embs, graph,  digraph, token_ids, pair, rel_emb, input_ids):
#         emb = emb.view(-1, dembs)
        """
        "  Based on the number of pairs, we want to create seperate copies to indicate different relations
        "  emb and hid size are num_pair*num_max_sentLen*demb
        """

        hid_f = self.hid_init_linear(embs)
        hid_f_res = hid_f
        
        for i in torch.arange(self.nlayers):
            hid_f, sp, ae, att = self.encoders[i](hid_f, graph, digraph,  token_ids, pair,rel_emb)
            hid_f = self.LayerNorm(hid_f+hid_f_res)
            hid_f_res = hid_f
            if i == self.nlayers - 1 or i == 0:
                attention_holder.append([sp, ae, att, pair, input_ids.cpu().data.numpy(), i])
#                 with open("../att_vis.pkl", "ab+") as f:
#                     pickle.dump([sp, ae, att, graph, pair, input_ids], f)
#                     print("saved current attention")

#         hid_final = hid_final.view(bs, -1, self.demb)
        return hid_f

class TransformerEncoder(nn.Module):

    def __init__(self, device, num_heads=1, dmodel=1024, demb=1024, dropout=0.1):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.dmodel = dmodel
        self.demb = demb
        #triplet for an edge representation
        self.link_rel_layer = nn.Linear(3*dmodel, dmodel)
        torch.nn.init.xavier_uniform(self.link_rel_layer.weight)
        #multihead attention, we need a forward multihead attention and a backward multihead_attention
        self.multihead_attention_forward = MultiHeadAttention(num_heads=num_heads, d_model=dmodel, d_k=int(dmodel/num_heads), device=device)
        self.multihead_attention_backward = MultiHeadAttention(num_heads=num_heads, d_model=dmodel, d_k=int(dmodel/num_heads), device=device)
    
        #fushion layer
        m = nn.Sigmoid()
        self.fushion = nn.Linear(2*dmodel, dmodel)
        torch.nn.init.xavier_uniform(self.fushion.weight)
        #final ffn sublayer
        self.ffn1 = nn.Linear(dmodel, 4*dmodel)
        torch.nn.init.xavier_uniform(self.ffn1.weight)
        self.ffn2 = nn.Linear(4*dmodel, 2*dmodel)
        torch.nn.init.xavier_uniform(self.ffn2.weight)
        #final representation layer
        self.repre = nn.Linear(2*dmodel, demb)
        torch.nn.init.xavier_uniform(self.repre.weight)
        self.layernorm = nn.LayerNorm(dmodel, eps=1e-12)
    
    """
    " create embeddings from the path
    """
    def create_path(self, graph, digraph, input_f, cur_tuple, rel_emb):
        src_list = set()

        srcNode = cur_tuple[0]
        tarNode = cur_tuple[1]
        if not graph.has_node(srcNode):
            srcNode -= 1
        if not graph.has_node(tarNode):
            tarNode -= 1
        shortest_path = nx.shortest_path(graph, source=srcNode, target=tarNode)

        """
        " Skip direct neighbors which will be added in the next step
        """
        all_edges = graph.edges(shortest_path)
        for pair in all_edges:
 
            if digraph.has_edge(pair[0], pair[1]):
                source = input_f[pair[0]]
                target = input_f[pair[1]]

                idx = digraph.get_edge_data(pair[0], pair[1])['dep']
                rel = rel_emb(torch.LongTensor([idx]).to(self.device)).view(-1)
                cur_link_f = torch.cat((source, rel, target))
            else:
                source = input_f[pair[1]]
                target = input_f[pair[0]]

                idx = digraph.get_edge_data(pair[1], pair[0])['dep']
                rel = rel_emb(torch.LongTensor([idx]).to(self.device)).view(-1)
                cur_link_f = torch.cat((source, rel, target))
            src_list.add(cur_link_f)  

        
        src_list = list(src_list)

        src_len = len(src_list)
 
        if len(src_list) > 0:
            src_list = torch.stack(src_list).to(device)

        return src_list, src_len, shortest_path, all_edges

    def create_links(self, graph, di_graph, input_f, rel_emb, cur_tuple):
    # to match with batch
        
        all_batch_forward = list()
        all_length_f = list()

        """
        " extract shortest path from the graph,construct embedding table for aggregation
        " 
        """
        node_len_f = list()
        init_tensor_f = torch.zeros(0).to(self.device)
        src_list, edges_src, sp, ae = self.create_path(graph, di_graph, input_f,cur_tuple, rel_emb)
        for node in sorted(list(di_graph.nodes())):

            edges_f = list()

            for cur in di_graph.out_edges(node, data=True):
                source = input_f[cur[0]]
                target = input_f[cur[1]]
                rel = rel_emb(torch.LongTensor([cur[2]['dep']]).to(self.device)).view(-1)

                cur_link_f = torch.cat((source, rel, target))
                edges_f.append(cur_link_f)

            for cur in di_graph.in_edges(node, data=True):
                source = input_f[cur[0]]
                target = input_f[cur[1]]
                rel = rel_emb(torch.LongTensor([cur[2]['dep']]).to(self.device)).view(-1)
                cur_link_f = torch.cat((source, rel, target)) 
                edges_f.append(cur_link_f)
            if len(edges_f) > 0:
                edges_f = torch.stack(edges_f).to(device)
                init_tensor_f = torch.cat((init_tensor_f, edges_f))

            node_len_f.append(len(edges_f))
#                     node_len_b.append(len(edges_b))
  
        return edges_f, torch.LongTensor(node_len_f).to(self.device), src_list, edges_src, sp,ae
    
    
    """
    " input_f bs*numsample*demb
    " input_b bs*numsample*demb
    """
    def forward(self, input_f, graph, di_graph, token_ids, pair, rel_emb):
#         print( graph.nodes(), di_graph.nodes(), token_ids, tuples)
        cur_graph, edge_f, path, path_len, sp, ae = self.create_links(graph, di_graph, input_f, rel_emb, pair)
        #hiddent states for triplets
        path = self.link_rel_layer(path)

        edg_emb = self.link_rel_layer(cur_graph)
        output_f, attentions = self.multihead_attention_forward(path, path_len, input_f, edg_emb, edg_emb,
                                                    graph, edge_f, token_ids, pair, rev=True)

        
        return output_f, sp, ae, attentions
    
"""
" To produce either forward or backward multihead temporal attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 d_k,
                 device,
                 num_heads=1,
                 dropout=0.1):
        super().__init__()
        assert d_k % num_heads == 0
        self.d_model = d_model
        self.d_k = d_k 
        self.num_heads = num_heads
        self.device = device
        self.q_pair_linear = nn.Linear(d_k*num_heads*2,d_model ,bias=False)
        self.path_linear_k = nn.Linear(d_k*num_heads, d_model,bias=False)
        self.path_linear_v = nn.Linear(d_k*num_heads, d_model,bias=False)
        self.q_linear = nn.Linear(d_k*num_heads,d_model ,bias=False)
        torch.nn.init.xavier_uniform(self.q_linear.weight)
        self.k_linear = nn.Linear(d_k*num_heads, d_model,bias=False)
        torch.nn.init.xavier_uniform(self.k_linear.weight)
        self.v_linear = nn.Linear(d_k*num_heads, d_model, bias=False)
        torch.nn.init.xavier_uniform(self.v_linear.weight)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform(self.out.weight)
        self.path_agg = nn.Linear(2*d_k*num_heads, d_model, bias=False)
    
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
    
    """
    " Dimension of q is batch*nsample*emb 
    " Dimension of k is batch*all_graph_edges which requires special handling
    """
    def forward(self, path, path_len, q, k, v, graphs, edge_len, token_ids, pair, rev):

        sen_len = q.size(0)
        returned = torch.clone(q)
        # perform linear operation and split into h heads
        #create concated representation for queries

        if rev == False:
            src = pair[0]
            tar = pair[1]
        else:
            src = pair[1]
            tar = pair[0]

        cancated_q = torch.cat((q[src], q[tar]))
        pair_q = cancated_q
        pair_q = self.q_pair_linear(pair_q)
        pair_q = pair_q.view(-1, self.num_heads, self.d_k).transpose(0,1)
        
        q = self.q_linear(q)
        q = q.view(-1, self.num_heads, self.d_k).transpose(0,1)
 

        new_path_k = self.path_linear_k(path).view(-1,self.num_heads, self.d_k).transpose(0,1)
        new_path_v = self.path_linear_v(path).view(-1,self.num_heads, self.d_k).transpose(0,1)
        new_k = self.k_linear(k).view(-1,self.num_heads, self.d_k).transpose(0,1)
        new_v = self.v_linear(v).view(-1,self.num_heads, self.d_k).transpose(0,1)
        # calculate attention using function we will define next

        new_hid_feature, path_feature, attentions = self.attention(
                                                    pair_q,
                                                    new_path_k,
                                                    new_path_v,
                                                    path_len,
                                                    q,              #nhead*dreduced*d_k
                                                    new_k,          #nhead*dreduced*d_k
                                                    new_v,
                                                    edge_len,       #dedges*dmodel
                                                    token_ids,
                                                    self.dropout,
                                                    self.device)
#         all_atts.append(tuple((attentions,cur_tuple, token_ids)))
        
        for idx, feature in enumerate(new_hid_feature):
#             print(feature.size(), path_feature.size())
            if token_ids[idx] == src:
                returned[token_ids[idx]] = self.path_agg(torch.cat((feature, path_feature)))
            elif token_ids[idx] == tar:
                returned[token_ids[idx]] = self.path_agg(torch.cat((path_feature, feature)))
            else:
                returned[token_ids[idx]] =  feature
        # concatenate heads and put through final linear layer
#         with open("../att_vis.pkl", "ab") as f:
#                 pickle.dump(all_atts, f)
#                 print("saved current attention")
        return returned,attentions
    

    from torch import nn
    def attention(self, pair_q, path, path_v, path_len, q, k,v, edges_len, token_ids, dropout=None, device=None):
        start = torch.LongTensor([0]).to(device)
        result_tensor = []
        softmax = nn.Softmax(dim=-1) 
        new_hid = []
        attention_list = list() 
        k_ = k.transpose(1,2)        #nhead*d_edge_len*k
        d_k = k_.size()[-1] ** 0.5

        """
        " First compute pair attention
        """
        path_result,att = self.scaled_dot_product(pair_q, path, path_v)
        path_result = torch.sum(path_result, dim=1)
#         path_result = torch.matmul(pair_q, path.transpose(1,2))                 #nhead*d_edge_len*dedge_len 
        
#         path_result = softmax(path_result/d_k)
#         print(path_result.size())
        attention_list.append(att)
        
#         path_result = torch.matmul(path_result, path)
#         path_result = path_result.transpose(0,1)
#         path_result = torch.sum(path_result, dim=0)

        path_result = self.out(path_result.flatten())
        """
        " Compute individual attention
        """

        for idx, i in enumerate(edges_len):
                                                             #nhead*d_edge_len*2k they are all the same
            
            if i == torch.LongTensor([1]).to(device):
                start += i
                continue
            q_ = q[:, token_ids[idx],:]                      #nhead*d_edge_len*k
            k_ = k[:,start:start+i,:]                        #nhead*d_edge_len*k
            v_ = v[:,start:start+i,:]
            d_k = k_.size()[-1] ** 0.5

            final_result,_ = self.scaled_dot_product(q_, k_, v_)

            final_result = torch.sum(final_result, dim=1)
            final_result = self.out(final_result.flatten())
#             raise
            if dropout is not None:
                final_result = dropout(final_result)
            new_hid.append(final_result)
            start += i

        new_hid = torch.stack(new_hid).to(self.device)
        
        return new_hid, path_result, attention_list
    

    


