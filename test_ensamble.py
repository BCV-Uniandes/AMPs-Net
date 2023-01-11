import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model.model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from dataset.dataset import AMPsDataset
from utils import metrics_pharma
import copy
import numpy as np 
import datetime
import os 
import csv 
import torch.nn.functional as F


@torch.no_grad()
def eval(model, device, loader, num_classes, args, target=None):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0 
    
    print('------Copying model 1---------')
    prop_predictor1 = copy.deepcopy(model)
    print('------Copying model 2---------')
    prop_predictor2 = copy.deepcopy(model)
    print('------Copying model 3---------')
    prop_predictor3 = copy.deepcopy(model)
    print('------Copying model 4---------')
    prop_predictor4 = copy.deepcopy(model)

    test_model_path = './log/'+args.save

    test_model_path1 = test_model_path+'/Fold1/model_ckpt/Checkpoint__valid_best.pth'
    test_model_path2 = test_model_path+'/Fold2/model_ckpt/Checkpoint__valid_best.pth'
    test_model_path3 = test_model_path+'/Fold3/model_ckpt/Checkpoint__valid_best.pth'
    test_model_path4 = test_model_path+'/Fold4/model_ckpt/Checkpoint__valid_best.pth'

        #LOAD MODELS
    print('------- Loading weights----------')
    prop_predictor1.load_state_dict(torch.load(test_model_path1,map_location=lambda storage, loc: storage)['model_state_dict'])
    prop_predictor1.to(device)
    
    prop_predictor2.load_state_dict(torch.load(test_model_path2,map_location=lambda storage, loc: storage)['model_state_dict'])
    prop_predictor2.to(device)
    
    prop_predictor3.load_state_dict(torch.load(test_model_path3,map_location=lambda storage, loc: storage)['model_state_dict'])
    prop_predictor3.to(device)
    
    prop_predictor4.load_state_dict(torch.load(test_model_path4,map_location=lambda storage, loc: storage)['model_state_dict'])
    prop_predictor4.to(device)
    
    #METHOD.EVAL
    prop_predictor1.eval()
    prop_predictor2.eval()
    prop_predictor3.eval()
    prop_predictor4.eval()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #breakpoint()
        batch = batch.to(device)
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            # only retain the top two node/edge features
            num_features = args.num_features
            batch.x = batch.x[:, :num_features]
            batch.edge_attr = batch.edge_attr[:, :num_features]

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):
                pred_logits1 = prop_predictor1(batch)
                pred_logits1 = F.softmax(pred_logits1,dim=1)

                pred_logits2 = prop_predictor2(batch)
                pred_logits2 = F.softmax(pred_logits2,dim=1)

                pred_logits3 = prop_predictor3(batch)
                pred_logits3 = F.softmax(pred_logits3,dim=1)

                pred_logits4 = prop_predictor4(batch)
                pred_logits4 = F.softmax(pred_logits4,dim=1)

                pred_logits = (pred_logits1+pred_logits2+pred_logits3+pred_logits4)/4     
                y_true.append(batch.y.view(batch.y.shape).detach().cpu())
                y_pred.append(pred_logits.detach().cpu())
                _, prediction_class = torch.max(pred_logits,1)
                if args.binary:
                    correct+=torch.sum(prediction_class == batch.y)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    if args.binary:
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
        nap, f = metrics_pharma.pltmap_bin(y_pred,y_true)
        acc = correct / len(loader.dataset)
        
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        map_metric, f_map = metrics_pharma.pltmap(y_pred,y_true,num_classes) 
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)
        
    if args.binary:
        return acc, auc, f, nap
    else:
        return auc, f, nap, map_metric, f_map['micro'], 

def main():

    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
        
    if args.binary:
        args.nclasses = 2
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    print(args)
    
    test_dataset = AMPsDataset(partition='Test',cross_val=None, binary_task=args.binary,args=args)       
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

    model = DeeperGCN(args).to(device)
    
    if args.binary:
        acc, auc, f, nap  = eval(model, device, test_loader, args.nclasses, args)
        
        save_items = {'Date': [], 'NAP': [], 'AUC': [], 'ACC': [], 'F_Med': []}        
        save_items["Date"] = datetime.date.today()
        save_items["NAP"] = nap
        save_items["AUC"] = auc
        save_items["ACC"] = acc.item()
        save_items["F_Med"] = f 
        fieldnames = list(save_items.keys())
        
    else:
        auc, f, nap, map_metric, f_map = eval(model, device, test_loader, args.nclasses, args)
        
        save_items = {'Date': [], 'NAP': [], 'F_Med': [], 'MAP':[], 'F_map':[], 'AUC': []}        
        save_items["Date"] = datetime.date.today()
        save_items["NAP"] = nap
        save_items["F_Med"] = f 
        save_items["MAP"] = map_metric
        save_items["F_map"] = f_map
        save_items["AUC"] = auc
        
        fieldnames = list(save_items.keys())
    
    csv_file = os.path.join('./log/',args.save,'Test_Ensamble.csv')

    with open(csv_file, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
            writer.writeheader()
            writer.writerow(save_items)
    
    if args.binary:
        print('NAP: {}, ACC: {}'.format(nap,acc))
    else:
        print('NAP: {}, MAP: {}, AUC: {}'.format(nap,map_metric,auc))


if __name__ == "__main__":
    main()
 

