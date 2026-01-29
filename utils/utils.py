import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#from sentence_transformers import SentenceTransformer
#from sentence_transformers.util import cos_sim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

def read_dict(dataset, file_name):

    data_dict = {}
    reverse_dict = {}


    max_value = 0
    entries = []

    with open(f"./data/{dataset}/{file_name}", "r") as file:
        for line in file:
            value, key = line.strip().split("\t")
            key_int = int(key)
            entries.append((key_int, value))
            data_dict[key_int] = value
            if key_int > max_value:
                max_value = key_int


    reverse_start = max_value
    for idx, (key, value) in enumerate(entries):
        reverse_value = f"{value}_reverse"
        reverse_key = reverse_start + key
        reverse_dict[reverse_key] = reverse_value


    data_dict.update(reverse_dict)

    return data_dict


    return data_dict

def get_words(input_data, entity_dict, rel_dict):
    result_list = []
    for row in input_data:
        if row[0] != -1:
            head = entity_dict.get(str(row[0]))
            rel = rel_dict.get(str(row[1]%len(rel_dict)))
            result_list.append(head+ " " + rel)
    return result_list


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms

def select_relevant_slices_v1(query, history_seq, entity_dict, rel_dict, k=64,alpha=0.01, mode="linear"):
    """
    query: 当前问题语义特征
    history_seq: 历史序列
    k: 选择的时间片数量
    """
    query_new = query.clone().float()
    query_new = torch.index_select(
	    query_new, 
	    dim=1,
	    index=torch.tensor([0, 1, 3], dtype=torch.long).to(query_new.device)
	)
    history_seq_new = history_seq.clone().float()
    history_seq_new = history_seq_new.permute(1, 0, 2)

    history_flat = history_seq_new.reshape(history_seq_new.size(0),-1)  # [B, M*N]
    query_flat = query_new.reshape(1, -1)


    cosine_sim = torch.nn.functional.cosine_similarity(history_flat, query_flat, dim=1)


    time_indices = torch.arange(history_seq_new.size(0), device=cosine_sim.device).float()
#    if mode == "linear":

#    elif mode == "exponential":

#    else:
#        raise ValueError("Unsupported mode. Use 'linear' or 'exponential'.")

    adjusted_sim = cosine_sim 
#    * time_weights

    topk_values, topk_indices = torch.topk(adjusted_sim, k=k)  # [582, 64]

    selected_seq = history_seq[:, topk_indices, :]
#    print(adjusted_sim)
#    print(topk_indices)
#    print(topk_values)
#    exit()
    return selected_seq


def tensor_to_text(tensor, entity_dict, rel_dict):
    words = []
    tensor_cpu = tensor.cpu().numpy()
    for row in tensor_cpu:
        col0, col1 = row[0], row[1]
        if col0 == -1 or col1 == -1:
            continue

        col0_int = int(col0)
        col1_int = int(col1)

        word0 = entity_dict.get(col0_int, "<UNK>")
        word1 = rel_dict.get(col1_int, "<UNK>")
        words.append(f"{word0} {word1}")
    return " ".join(words)

def call_similarity(sentences1, sentences2):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('./models/all-MiniLM-L6-v2', device=device)


    embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)



    cosine_scores = cos_sim(embeddings1, embeddings2)
    return cosine_scores
    
def select_relevant_slices(query, history_seq, entity_dict, rel_dict, k=64):
    """
    query: 当前问题语义特征
    history_seq: 历史序列
    k: 选择的时间片数量
    """

    
    query_new = query.clone().int()
    query_new = torch.index_select(
	    query_new, 
	    dim=1,
	    index=torch.tensor([0, 1, 3], dtype=torch.long).to(query_new.device)
	)
    history_seq_new = history_seq.clone().int()
#    print(history_seq_new[0])
#    print(history_seq_new[0].shape)
    similarity = []
    history_norm = history_seq_new.permute(1, 0, 2)
    text0 = tensor_to_text(query_new, entity_dict, rel_dict)
    for Slice in history_norm:
        text1 = tensor_to_text(Slice, entity_dict, rel_dict)
        similarity.append(call_similarity(text0, text1))
    exit()
#    query_norm = F.normalize(query_new, p=2, dim=-1)  # [582, 200]
#    history_norm = F.normalize(history_seq_new, p=2, dim=-1)  # [582, 128, 200]
#    
#

#    
#    similarity = torch.einsum('bd,btd->bt', query_norm, history_norm)  # [582, 128]


    beta = 5.0
    seq_len = history_seq.size(1)
    time_indices = torch.arange(seq_len, device=query.device)  # [0,1,2...,127]
    delta_t = (seq_len - 1 - time_indices).float()
    time_weight = torch.exp(-beta * delta_t)
    

    adjusted_sim = similarity * time_weight.unsqueeze(0)  # [582,128] * [1,128]


    topk_values, topk_indices = torch.topk(adjusted_sim, k=k, dim=1)  # [582, 64]

    


    selected_seq = torch.gather(history_seq, 1, 
                               topk_indices.unsqueeze(-1).expand(-1, -1, 3)) # [582,64,200]

#    print(history_seq[:,0,:])
#    print(history_seq)
#    for i in range(selected_seq.size(1)):

#        sub_tensor = selected_seq[:, i, :]
#    

#        sorted_indices = torch.argsort(sub_tensor[:, 2])
#    

#        selected_seq[:, i, :] = sub_tensor[sorted_indices]
#    print(history_seq[:,0,:])
#    print(query)
#    print(selected_seq)
#    print(history_seq.shape)
#    print(query.shape)
    return selected_seq
