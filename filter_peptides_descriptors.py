import pandas as pd 
import math
import numpy as np
import os 
from tqdm import tqdm
import csv 
import json
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import argparse


def select_higher_probs(args):

    path_file = os.path.join(args.path_to_files, args.file_name)
    threashold = args.threashold
    
    inference_results = pd.read_csv(path_file)
    inference_results.sort_values(by=['AMP score'], inplace=True, ascending=False)
    
    selected_peptides = inference_results[inference_results['AMP score']>=threashold]
    selected_peptides.reset_index(inplace=True, drop=True)
    path_file_new = path_file.split('.')[1] + '_High_Probs.csv'
    path_file = '.' + os.path.join(args.path_to_files, path_file_new)
    selected_peptides.to_csv(path_file, index=False)

    return selected_peptides

def select_by_descriptors(args, selected_peptides):

    path_file = os.path.join(args.path_to_files, args.file_name)

    inference_results = selected_peptides
    inference_results['Size'] = ""
    inference_results['BomanI'] = ""
    inference_results['NetCharge'] = ""
    inference_results['HydrophobicRatio'] = ""
    inference_results['HydrophobicMoment'] = ""
    inference_results['Aliphatic'] = ""
    inference_results['InstaIndex'] = ""
    inference_results['IsoelectricPoint'] = ""
    drop_idx = []
    
    for idx in tqdm(range(len(inference_results))):
        peptide = inference_results.Sequence.iloc[idx]
        #Size
        size = len(peptide)
        inference_results['Size'][idx] = size
        if size < 10 or size > 30:
            drop_idx.append(idx)
            continue
        #Descriptors
        desc = GlobalDescriptor(peptide)
        desc2 = PeptideDescriptor(peptide,'eisenberg')
        #BomanIndex
        desc.boman_index()
        bomanIndex = desc.descriptor[0][0]
        inference_results['BomanI'][idx] = bomanIndex
        #HydrophobicRatio
        desc.hydrophobic_ratio()
        HR = desc.descriptor[0][0]
        inference_results['HydrophobicRatio'][idx] = HR
        if HR < 0.5:
            drop_idx.append(idx)
            continue
        #Aliphatic 
        desc.aliphatic_index()
        al = desc.descriptor[0][0] 
        inference_results['Aliphatic'][idx] = al
        #Instability Index
        desc.instability_index()
        II = desc.descriptor[0][0]
        inference_results['InstaIndex'][idx] = II
        if II > 40: 
            drop_idx.append(idx)
            continue
        #Isoelectric Point
        desc.isoelectric_point()
        IP = desc.descriptor[0][0]
        inference_results['IsoelectricPoint'][idx] = IP
        if IP > 20 or IP < 0:
            drop_idx.append(idx)
            continue
        #HydrophobicMoment
        desc2.calculate_moment(window=1000)
        HM = desc2.descriptor[0][0] 
        inference_results['HydrophobicMoment'][idx] = HM
        if HM > 0.3:
            drop_idx.append(idx)
            continue
        #NetCharge
        desc.calculate_charge(ph=7.4)
        netcharge = desc.descriptor[0][0]
        inference_results['NetCharge'][idx] = netcharge
        if netcharge <= 0:
            drop_idx.append(idx) 
            continue

    for idx in drop_idx:
        inference_results.drop(idx, inplace=True)
                 
    inference_results.sort_values(by=['BomanI','NetCharge','Aliphatic'], inplace=True, ascending=False)
    inference_results.reset_index(inplace=True, drop=True)
    path_file_new = path_file.split('.')[1] + '_Select_Descriptors.csv'

    path_file = '.' + os.path.join(args.path_to_files, path_file_new)
    inference_results.to_csv(path_file, index=False)

def main(args):

    selected_peptides = select_higher_probs(args)
    select_by_descriptors(args,selected_peptides)

    
if __name__ == "__main__":   

    parser = argparse.ArgumentParser(description="Generate metadata.")
    parser.add_argument("--path_to_files", type=str, default='./Inference/AMPs/')
    parser.add_argument("--file_name", type=str, default='Example.csv')
    parser.add_argument("--threashold", type=float, default=0.95)
    args = parser.parse_args()

    main(args)