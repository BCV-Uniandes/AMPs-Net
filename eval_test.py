import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model.model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from dataset.dataset import AMPsDataset
import torch.nn.functional as F
from utils import metrics_pharma
import copy
import numpy as np 
import os 
import csv 



@torch.no_grad()
def eval(model, device, loader, num_classes, cross_val, args):
    model.eval()
    y_true = []
    y_pred = []
    
    print('------Copying model {}---------'.format(cross_val))
    prop_predictor = copy.deepcopy(model)

    test_model_path = './log/'+args.save
    test_model_path = test_model_path+'/Fold{}/model_ckpt/Checkpoint__valid_best.pth'.format(cross_val)
    
    #LOAD MODELS
    print('------- Loading weights----------')
    prop_predictor.load_state_dict(torch.load(test_model_path,map_location=lambda storage, loc: storage)['model_state_dict'])
    prop_predictor.to(device)
   
    #METHOD.EVAL
    prop_predictor.eval()


    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        batch_mol = batch.to(device)
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            # only retain the top two node/edge features
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):   

                pred = F.softmax(prop_predictor(batch_mol),dim=1)
                    
                
                y_true.append(batch_mol.y.view(batch_mol.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if args.binary:
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
        nap, f = metrics_pharma.pltmap_bin(y_pred,y_true)     
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    return nap
    
def main():

    args = ArgsInit().args
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
        
    if args.binary:
        args.nclasses = 2
 
    #Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    print(args)
    maps = []
    
    for cross_val in range(1,5):
        
        
        test_dataset = AMPsDataset(partition='Test', cross_val=cross_val, binary_task=args.binary,args=args)
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers)


        model = DeeperGCN(args).to(device)
        maps.append(eval(model, device, test_loader, args.nclasses, cross_val, args))

    map1, map2, map3, map4 = maps
    save_items = {'Mean': [], 'Fold1': [], 'Fold2': [], 'Fold3': [], 'Fold4': []}
    
    mean_map = np.mean([map1,map2,map3,map4])
    
    save_items['Mean'] = mean_map
    save_items["Fold1"] = map1
    save_items["Fold2"] = map2
    save_items["Fold3"] = map3
    save_items["Fold4"] = map4
    


    fieldnames = list(save_items.keys())
    
    csv_file = os.path.join('./log/',args.save,'Test.csv')

    with open(csv_file, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
            writer.writeheader()
            writer.writerow(save_items)


    print({'Mean' : mean_map,
           'Fold1': map1,
           'Fold2': map2,
           'Fold3': map3,
           'Fold4': map4})
    
if __name__ == "__main__":
    main()
