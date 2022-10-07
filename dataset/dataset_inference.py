import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
import csv
from tqdm import tqdm
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector) 

def read_file(name_file):
    All_data = {'Sequence': []}
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            All_data['Sequence'].append(row['Translation'])
    csv_file.close()
    data_file = pd.DataFrame.from_dict(All_data)
    return data_file

def read_file_metadata(name_file):
    All_data = {'Sequence': [], 'Size':[],'BomanI':[], 'NetCharge':[] ,'HydrophobicRatio':[] ,'HydrophobicMoment':[], 'Aliphatic':[], 'InstaIndex':[], 'IsoelectricPoint':[]}
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                All_data['Sequence'].append(row['Sequence'])
                All_data['Size'].append(row['Size'])
                All_data['BomanI'].append(row['BomanI'])
                All_data['NetCharge'].append(row['NetCharge'])
                All_data['HydrophobicRatio'].append(row['HydrophobicRatio'])
                All_data['HydrophobicMoment'].append(row['HydrophobicMoment'])
                All_data['Aliphatic'].append(row['Aliphatic'])
                All_data['InstaIndex'].append(row['InstaIndex'])
                All_data['IsoelectricPoint'].append(row['IsoelectricPoint'])
            except:
                print('Missing some metadata')
                breakpoint()
    csv_file.close()
    data_file = pd.DataFrame.from_dict(All_data)
    return data_file


def load_dataset(cross_val,binary_task,file_inference):
    
    print('Loading data...')
    path='./data/datasets/Inference/'
    Inference = read_file_metadata(path + file_inference)

    return Inference

def smiles_to_graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromFASTA(smiles_string)
    if mol == None:
        print(smiles_string)
        breakpoint()

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        ftrs = atom_to_feature_vector(atom)
        atom_features_list.append(ftrs)
            
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return edge_attr, edge_index, x 


def transform_molecule_pg(mol,label,args,metadata=None):


    edge_attr, edge_index, x = smiles_to_graph(mol)
    
    x = torch.tensor(x)
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    y = [label]
    if args.metadata:
        metadata = torch.tensor([metadata])
        return Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, metadata=metadata)
    else:
        return Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y)


def load_data(partition,cross_val,binary_task,file_inference):
    
    Inference = load_dataset(cross_val,binary_task,file_inference)
    
    return Inference
        
class AMPsDataset(Dataset):
    
    """Dataset for Antimicrobial Peptides."""

    def __init__(self, partition, cross_val, binary_task, file_inference,args):
        """
        Args:
            smiles_dir (string): Directory with all the peptides' FASTAs.
        """
        self.peptides = load_data(partition,cross_val,binary_task,file_inference)
        self.arguments = args
        
    def __len__(self):
        
        return len(self.peptides)
    
    def __getitem__(self,idx):
        
        metadata = None
        
        if self.arguments.metadata:
            metadata_criteria = ['Size','BomanI','NetCharge','HydrophobicRatio','HydrophobicMoment','Aliphatic','InstaIndex','IsoelectricPoint']
            if self.arguments.num_metadata < len(metadata_criteria):
                for description in self.arguments.delete_descriptor:
                    metadata_criteria.remove(description)
            metadata=[]
            for selection_criteria in metadata_criteria:
                try:
                    metadata.append(float(self.peptides[selection_criteria].iloc[idx]))
                except:
                    print('Missing some of metadata')
                    breakpoint()

        mol=self.peptides.Sequence.iloc[idx]
        label=self.peptides.Sequence.iloc[idx]
        peptide_graph = transform_molecule_pg(mol,label,self.arguments,metadata)
        
        return peptide_graph

    