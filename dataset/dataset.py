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

def read_file(name_file,args):
    if args.metadata:
        All_data = {'Sequence': [], 'Label': [], 'AM-Activity':[], 'Size':[],'BomanI':[], 'NetCharge':[] ,'HydrophobicRatio':[] ,'HydrophobicMoment':[], 'Aliphatic':[], 'InstaIndex':[], 'IsoelectricPoint':[]}
    else:
        All_data = {'Sequence': [], 'Label': [], 'AM-Activity':[]}
    
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                All_data['Sequence'].append(row['Sequence'])
                All_data['Label'].append(int(row['Label']))
                All_data['AM-Activity'].append(row['Activity'])
                if args.metadata:
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

def read_multilabel_file(name_file,args):
    if args.metadata:
        All_data = {'Sequence': [], 'Antibacterial':[], 'Antiviral':[], 'Antiparasitic':[], 'Antifungal':[], 'Activity_Label':[], 'Size':[],'BomanI':[], 'NetCharge':[] ,'HydrophobicRatio':[] ,'HydrophobicMoment':[], 'Aliphatic':[], 'InstaIndex':[], 'IsoelectricPoint':[]}
    else:
        All_data = {'Sequence': [], 'Antibacterial':[], 'Antiviral':[], 'Antiparasitic':[], 'Antifungal':[], 'Activity_Label':[]}
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            All_data['Sequence'].append(row['Sequence'])
            All_data['Antibacterial'].append(int(row['Antibacterial']))
            All_data['Antiviral'].append(int(row['Antiviral']))
            All_data['Antiparasitic'].append(int(row['Antiparasitic']))
            All_data['Antifungal'].append(int(row['Antifungal']))
            All_data['Activity_Label'].append(int(row['Activity_Label']))
            if args.metadata:
                All_data['Size'].append(row['Size'])
                All_data['BomanI'].append(row['BomanI'])
                All_data['NetCharge'].append(row['NetCharge'])
                All_data['HydrophobicRatio'].append(row['HydrophobicRatio'])
                All_data['HydrophobicMoment'].append(row['HydrophobicMoment'])
                All_data['Aliphatic'].append(row['Aliphatic'])
                All_data['InstaIndex'].append(row['InstaIndex'])
                All_data['IsoelectricPoint'].append(row['IsoelectricPoint'])
    csv_file.close()
    data_file = pd.DataFrame.from_dict(All_data)
    return data_file

def read_allclasses_file(name_file,args):
    names = ["Sequence","Antimicrobial","Antibacterial","Antiviral",
             "Neuropeptide", "unknown","Signal-Peptide","Immunomodulating",
             "Antifungal","Anuran-Defense","Anticancer","Cell-Penetrating",
             "ACE-Inhibitor","Antioxidative", "TumorHoming","Toxic","Peptidase-Inhibitor",
             "AntiTubercular","Antiparasitic","QuorumSensing-Peptide","Opioid","BBB-Peptide",
             "antihypertensive","Haemolytic","Antiamnestic","Antithrombotic","CaMKII-Inhibitor"]
    All_data = {k: [] for k in names}
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            for name in names:
                if name == 'Sequence':
                    All_data[name].append(row[name])
                else:     
                    All_data[name].append(int(row[name]))
    csv_file.close()
    data_file = pd.DataFrame.from_dict(All_data)
    return data_file



def load_dataset(cross_val,binary_task,args):
    print('Loading data...')
    
    if binary_task:
        path='./data/datasets/AMP/'
        print(path)
        A = read_file(path + "Fold1.csv",args)
        B = read_file(path + "Fold2.csv",args)
        C = read_file(path + "Fold3.csv",args)
        D = read_file(path + "Fold4.csv",args)
        data_test = read_file(path + "Test.csv",args)
    elif args.multilabel:
        path='./data/datasets/MultiLabel/'
        print(path)
        A = read_multilabel_file(path + "Fold1.csv",args)
        B = read_multilabel_file(path + "Fold2.csv",args)
        C = read_multilabel_file(path + "Fold3.csv",args)
        D = read_multilabel_file(path + "Fold4.csv",args)
        data_test = read_multilabel_file(path + "Test.csv",args)        
        
    if cross_val == 1:
        data_train = pd.concat([A, B, C], ignore_index=True)
        data_val = D
    elif cross_val == 2:
        data_train = pd.concat([A, C, D], ignore_index=True)
        data_val = B
    elif cross_val == 3:
        data_train = pd.concat([A, B, D], ignore_index=True)
        data_val = C
    elif cross_val == 4:
        data_train = pd.concat([B, C, D], ignore_index=True)
        data_val = A 
    else:
        print('No valid partition')     
 
    if cross_val != None:
        return data_train, data_val, data_test
    else:
        return None, None, data_test

def smiles_to_graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromFASTA(smiles_string)
    if mol == None:
        print(smiles_string)
        print('Invalid smile')
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

def get_dataset(dataset):
    total_dataset = []

    for mol,label in tqdm(zip(dataset['Sequence'],dataset['Label']),total=len(dataset['Sequence'])):
       
        total_dataset.append(transform_molecule_pg(mol,label))
    return total_dataset

def transform_molecule_pg(mol,label,args,metadata=None):
    

    edge_attr, edge_index, x = smiles_to_graph(mol)
    
    x = torch.tensor(x)
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    
    if args.multilabel:
        y = torch.tensor([label])
    else:
        y = torch.tensor([int(label)])
        
    if args.metadata:
        metadata = torch.tensor([metadata])
        return Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, metadata=metadata)
    else:
        return Data(edge_attr=edge_attr, edge_index=edge_index, x=x, y=y)

def load_data(partition,cross_val,binary_task,args):
    
    train, val, test = load_dataset(cross_val,binary_task,args)
    
    if partition=='Train':
        return train
    elif partition=='Val':
        return val
    elif partition=='Test':
        return test
        

class AMPsDataset(Dataset):
    
    """Dataset for Antimicrobial Peptides."""

    def __init__(self, partition, cross_val, binary_task,args):
        """
        Args:
            smiles_dir (string): Directory with all the peptides' FASTAs.
        """
        self.peptides = load_data(partition,cross_val,binary_task,args)
        self.arguments = args
        
    def __len__(self):
        
        return len(self.peptides)
    
    def __getitem__(self,idx):

        metadata = None
        
        if self.arguments.metadata:
            metadata_criteria = ['Size','BomanI','NetCharge','HydrophobicRatio','HydrophobicMoment','Aliphatic','InstaIndex','IsoelectricPoint']
            if self.arguments.num_metadata < len(metadata_criteria):
                for description in self.arguments.delete_descriptor:
                    try:
                        metadata_criteria.remove(description)
                    except:
                        print(f'Unable to remove metadata because {description} does not exists.')
                        breakpoint()
            metadata=[]
            for selection_criteria in metadata_criteria:
                try:
                    metadata.append(float(self.peptides[selection_criteria].iloc[idx]))
                except:
                    print('Missing some metadata')
                    breakpoint()
        if self.arguments.multilabel:
            mol=self.peptides.Sequence.iloc[idx]
            AB=self.peptides.Antibacterial.iloc[idx]
            AV=self.peptides.Antiviral.iloc[idx]
            AF=self.peptides.Antifungal.iloc[idx]
            AP=self.peptides.Antiparasitic.iloc[idx]
            label=[int(AB),int(AV),int(AF),int(AP)]
            peptide_graph = transform_molecule_pg(mol,label,self.arguments,metadata)        
        else:  
            mol=self.peptides.Sequence.iloc[idx]
            label=self.peptides.Label.iloc[idx]
            peptide_graph = transform_molecule_pg(mol,label,self.arguments,metadata)
        
        return peptide_graph

    