import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import pandas as pd

def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
    q_to_sro = set()
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])

    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])


    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'})
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values})


    dst_df = df_dic.query('sr in @er_list')

    
    dst_get = dst_df['dst'].values
    two_ent = set().union(*dst_get)
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')
    dst_get_inv = dst_df_inv['dst'].values
    two_ent_inv = set().union(*dst_get_inv)
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')

    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    return  q_tri,q_tri_inv


def update_dict(subg_arr, s_to_sro, sr_to_sro,num_rels):

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]

        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])

    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

dataset_list = ["ICEWS14", "ICEWS18","ICEWS05-15"]


for dataset in dataset_list:
    train_data, train_times = load_quadruples('./{}'.format(dataset), 'train.txt')
    num_nodes,num_rels= get_total_number('./{}'.format(dataset), 'stat.txt')
    print("the number of entity and relation", num_nodes,num_rels)

    train_list = split_by_time(train_data)
    id_list = [_ for _ in range(len(train_list))]
    sample_len = 3

    save_dir_subg = './{}/his_graph_for/'.format(dataset)
    save_dir_obj = './{}/his_graph_inv/'.format(dataset)
    save_dir_sub = './{}/his_dict/'.format(dataset)

    def mkdirs(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdirs(save_dir_obj)
    mkdirs(save_dir_sub)
    mkdirs(save_dir_subg)


    sr_to_sro = defaultdict(set)
    s_to_sro = defaultdict(set)
    sro_to_fre = dict()
    subgraph_arr = []
    subgraph_arr_inv = []
    print("------------{}sample history graph-------------------------------------".format(dataset))
    all_list= train_list
    idx = [_ for _ in range(len(all_list))]
    for train_sample_num in tqdm(idx):
        if train_sample_num == 0: continue
        output = all_list[train_sample_num:train_sample_num+1]
        history_graph = all_list[train_sample_num-1:train_sample_num]
        update_dict(history_graph[0], s_to_sro, sr_to_sro, num_rels)
        if train_sample_num > 0:
            his_list = all_list[:train_sample_num]
            subg_arr = np.concatenate(his_list)
            sub_snap,sub_snap_inv = get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, output[0], num_nodes,num_rels)
        np.save('./{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
        np.save('./{}/his_graph_inv/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)
    np.save('./{}/his_dict/train_s_r.npy'.format(dataset), sr_to_sro)


    

