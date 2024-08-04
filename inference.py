import torch
from torch_geometric.loader import DataLoader
from model.model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from dataset.dataset_inference import AMPsDataset
import copy
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os

def get_device(args):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.device < device_count:
            device = torch.device(f"cuda:{args.device}")
        else:
            print(f"Warning: Specified CUDA device {args.device} is not available. Using device 0 instead.")
            device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    return device


@torch.no_grad()
def eval(model, device, loader, num_classes, args, target=None):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0

    print("------Copying model 1---------")
    prop_predictor1 = copy.deepcopy(model)
    print("------Copying model 2---------")
    prop_predictor2 = copy.deepcopy(model)
    print("------Copying model 3---------")
    prop_predictor3 = copy.deepcopy(model)
    print("------Copying model 4---------")
    prop_predictor4 = copy.deepcopy(model)

    test_model_path = args.save

    test_model_path1 = test_model_path + "/Fold1/model_ckpt/Checkpoint.pth"
    test_model_path2 = test_model_path + "/Fold2/model_ckpt/Checkpoint.pth"
    test_model_path3 = test_model_path + "/Fold3/model_ckpt/Checkpoint.pth"
    test_model_path4 = test_model_path + "/Fold4/model_ckpt/Checkpoint.pth"

    # LOAD MODELS
    print("------- Loading weights----------")
    prop_predictor1.load_state_dict(torch.load(test_model_path1, map_location='cuda:0')["model_state_dict"])
    prop_predictor1.to(device)

    prop_predictor2.load_state_dict(torch.load(test_model_path2, map_location='cuda:0')["model_state_dict"])
    prop_predictor2.to(device)

    prop_predictor3.load_state_dict(torch.load(test_model_path3, map_location='cuda:0')["model_state_dict"])
    prop_predictor3.to(device)

    prop_predictor4.load_state_dict(torch.load(test_model_path4, map_location='cuda:0')["model_state_dict"])
    prop_predictor4.to(device)

    # METHOD.EVAL
    prop_predictor1.eval()
    prop_predictor2.eval()
    prop_predictor3.eval()
    prop_predictor4.eval()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # breakpoint()
        batch = batch.to(device)
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            # only retain the top two node/edge features
            num_features = args.num_features
            batch.x = batch.x[:, :num_features]
            batch.edge_attr = batch.edge_attr[:, :num_features]

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):

                pred_logits1 = prop_predictor1(batch)
                pred_logits2 = prop_predictor2(batch)
                pred_logits3 = prop_predictor3(batch)
                pred_logits4 = prop_predictor4(batch)

                if args.multilabel:
                    pred_logits1 = F.sigmoid(pred_logits1)
                    pred_logits2 = F.sigmoid(pred_logits2)
                    pred_logits3 = F.sigmoid(pred_logits3)
                    pred_logits4 = F.sigmoid(pred_logits4)
                else:
                    pred_logits1 = F.softmax(pred_logits1, dim=1)
                    pred_logits2 = F.softmax(pred_logits2, dim=1)
                    pred_logits3 = F.softmax(pred_logits3, dim=1)
                    pred_logits4 = F.softmax(pred_logits4, dim=1)

                pred_logits = (
                    pred_logits1 + pred_logits2 + pred_logits3 + pred_logits4
                ) / 4
                y_true.extend(batch.y)
                y_pred.append(pred_logits.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()

    return y_true, y_pred


def main():

    args = ArgsInit().args

    device = get_device(args)
    print(f"Using device: {device}")

    if args.binary:
        args.nclasses = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    print(args)

    test_dataset = AMPsDataset(
        partition="Inference",
        cross_val=None,
        binary_task=args.binary,
        file_inference=args.file_infe,
        args=args,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = DeeperGCN(args).to(device)

    sequence, score = eval(model, device, test_loader, args.nclasses, args)

    if args.binary:
        save_item = {"Sequence": [], "Non-AMP score": [], "AMP score": []}

        for idx in range(len(sequence)):
            save_item["Sequence"].append(sequence[idx][0])
            save_item["Non-AMP score"].append(score[idx][0])
            save_item["AMP score"].append(score[idx][1])

    elif args.multilabel:
        save_item = {
            "Sequence": [],
            "AB score": [],
            "AV score": [],
            "AP score": [],
            "AF score": [],
        }
        for idx in range(len(sequence)):
            save_item["Sequence"].append(sequence[idx][0])
            save_item["AB score"].append(score[idx][0])
            save_item["AV score"].append(score[idx][1])
            save_item["AP score"].append(score[idx][2])
            save_item["AF score"].append(score[idx][3])

    inference_results = pd.DataFrame.from_dict(save_item)

    if args.binary:
        saving_dir = "Inference/AMPs/"
    elif args.multilabel:
        saving_dir = "Inference/MultiLabel/"

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir, exist_ok=True)
    path_results = saving_dir + args.file_infe
    inference_results.to_csv(path_results, index=False)


if __name__ == "__main__":
    main()
