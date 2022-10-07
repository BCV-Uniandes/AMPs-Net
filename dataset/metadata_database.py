import pandas as pd
import math
import numpy as np
import os
from tqdm import tqdm
import csv
import json
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import argparse
import __init__

def read_file_infe(name_file):
    All_data = {"Sequence": []}
    with open(name_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            All_data["Sequence"].append(row["Sequence"])
    csv_file.close()
    data_file = pd.DataFrame.from_dict(All_data)
    return data_file


def main(args):

    files = args.files
    path_to_files = args.path_to_files

    path_to_save = args.path_to_save

    for file_name in tqdm(files):
        path_file = os.path.join(path_to_files, file_name)
        inference_results = read_file_infe(path_file)
        inference_results["Size"] = ""
        inference_results["BomanI"] = ""
        inference_results["NetCharge"] = ""
        inference_results["HydrophobicRatio"] = ""
        inference_results["HydrophobicMoment"] = ""
        inference_results["Aliphatic"] = ""
        inference_results["InstaIndex"] = ""
        inference_results["IsoelectricPoint"] = ""

        for idx in tqdm(range(len(inference_results))):
            peptide = inference_results.Sequence.iloc[idx]
            # Size
            peptide = peptide.upper()
            size = len(peptide)
            if size == 0:
                continue
            inference_results["Size"][idx] = size
            # Descriptors
            desc = GlobalDescriptor(peptide)
            desc2 = PeptideDescriptor(peptide, "eisenberg")
            # BomanIndex
            try:
                desc.boman_index()
            except:
                breakpoint()
            bomanIndex = desc.descriptor[0][0]
            inference_results["BomanI"][idx] = bomanIndex
            # HydrophobicRatio
            desc.hydrophobic_ratio()
            HR = desc.descriptor[0][0]
            inference_results["HydrophobicRatio"][idx] = HR
            # Aliphatic
            desc.aliphatic_index()
            al = desc.descriptor[0][0]
            inference_results["Aliphatic"][idx] = al
            # Instability Index
            desc.instability_index()
            II = desc.descriptor[0][0]
            inference_results["InstaIndex"][idx] = II
            # Isoelectric Point
            desc.isoelectric_point()
            IP = desc.descriptor[0][0]
            inference_results["IsoelectricPoint"][idx] = IP
            # HydrophobicMoment
            desc2.calculate_moment(window=1000)
            HM = desc2.descriptor[0][0]
            inference_results["HydrophobicMoment"][idx] = HM
            # NetCharge
            desc.calculate_charge(ph=7.4)
            netcharge = desc.descriptor[0][0]
            inference_results["NetCharge"][idx] = netcharge
        new_path = os.path.join(path_to_save, file_name)
        inference_results.to_csv(new_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate metadata.")
    parser.add_argument("--files", nargs="+", default=[None],
                        help='File to generate metadata to')
    parser.add_argument("--path_to_files", type=str, default='./data/datasets/Inference')
    parser.add_argument("--path_to_save", type=str, default='./data/datasets/Inference')

    args = parser.parse_args()

    main(args)
