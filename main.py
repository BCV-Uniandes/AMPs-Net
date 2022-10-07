import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model.model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
import logging
import time
from dataset.dataset import load_dataset, AMPsDataset
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
time.sleep(3)
from utils import metrics_pharma
import sys


def train(model, device, loader, optimizer, num_classes, args):
    loss_list = []
    y_true = []
    y_pred = []
    correct = 0
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            # only retain the top two node/edge features
            num_features = args.num_features
            batch.x = batch.x[:, :num_features]
            batch.edge_attr = batch.edge_attr[:, :num_features]
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
            loss = 0
            if not args.binary:
                for i in range(0, num_classes):
                    class_mask = batch.y.clone()
                    class_loss = cls_criterion(
                        F.sigmoid(pred[:, i]).to(torch.float32),
                        class_mask.to(torch.float32),
                    )
                    loss += class_loss
            else:
                class_loss = cls_criterion(
                    F.sigmoid(pred[:, 1]).to(torch.float32), batch.y.to(torch.float32)
                )
                loss += class_loss

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pred = F.softmax(pred, dim=1)
            y_true.append(batch.y.view(batch.y.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            _, prediction_class = torch.max(pred, 1)
            correct += torch.sum(prediction_class == batch.y)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if args.binary:
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true)
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    acc = correct / len(loader.dataset)

    return acc, auc, f, nap, np.mean(loss_list)


@torch.no_grad()
def eval_gcn(model, device, loader, num_classes, args):
    model.eval()
    y_true = []
    y_pred = []
    loss_list = []
    correct = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
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
                pred = model(batch)
                loss = 0
                if not args.binary:
                    for i in range(0, num_classes):
                        class_mask = batch.y.clone()
                        class_mask[batch.y == i] = 1
                        class_mask[batch.y != i] = 0
                        class_loss = cls_criterion(
                            F.sigmoid(pred[:, i]).to(torch.float32),
                            class_mask.to(torch.float32),
                        )
                        loss += class_loss
                else:
                    class_loss = cls_criterion(
                        F.sigmoid(pred[:, 1]).to(torch.float32),
                        batch.y.to(torch.float32),
                    )
                    loss += class_loss

                loss_list.append(loss.item())
                pred = F.softmax(pred, dim=1)
                y_true.append(batch.y.view(batch.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                _, prediction_class = torch.max(pred, 1)
                correct += torch.sum(prediction_class == batch.y)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if args.binary:
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true)
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    acc = correct / len(loader.dataset)
    return acc, auc, f, nap, np.mean(loss_list)


def make_weights_for_balanced_classes(data, nclasses):
    count = [0] * nclasses
    for item in data:
        count[item] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val]

    return weight


def main():
    # Load arguments

    args = ArgsInit().save_exp()
    if args.use_gpu:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")
    # Change classes if binary task
    if args.binary:
        args.nclasses = 2
    # Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    sub_dir = "Checkpoint_"

    logging.info("%s" % args)
    # Load dataset

    data_train, _, _ = load_dataset(
        cross_val=args.cross_val, binary_task=args.binary, args=args
    )

    train_dataset = AMPsDataset(
        partition="Train", cross_val=args.cross_val, binary_task=args.binary, args=args
    )
    valid_dataset = AMPsDataset(
        partition="Val", cross_val=args.cross_val, binary_task=args.binary, args=args
    )

    if args.balanced_loader:
        # TRAIN WEIGTH
        weights_train = make_weights_for_balanced_classes(
            list(data_train.Label), args.nclasses
        )
        weights_train = torch.DoubleTensor(weights_train)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, len(weights_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
        )
    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = DeeperGCN(args).to(device)

    logging.info(model)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    results = {
        "Lowest_valid_loss": 10000,
        "highest_valid": 0,
        "highest_train": 0,
        "epoch": 0,
    }

    start_time = time.time()
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_nap = []
    val_epoch_nap = []

    if args.resume:

        model_name = os.path.join(args.save, "model_ckpt", args.model_load_path)
        assert os.path.exists(model_name), "Model checkpoint does not exist"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"] + 1
        train_epoch_loss = checkpoint["loss_train"]
        val_epoch_loss = checkpoint["loss_val"]
        train_epoch_nap = checkpoint["nap_train"]
        val_epoch_nap = checkpoint["nap_val"]
        print("Model loaded")
    else:
        init_epoch = 1

    loss_track = 0
    past_loss = 0

    for epoch in range(init_epoch, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info("Training...")

        tr_acc, tr_auc, tr_f, tr_nap, epoch_loss = train(
            model, device, train_loader, optimizer, args.nclasses, args
        )

        logging.info("Evaluating...")
        val_acc, val_auc, val_f, val_nap, val_loss = eval_gcn(
            model, device, valid_loader, args.nclasses, args
        )

        train_epoch_loss.append(epoch_loss)
        val_epoch_loss.append(val_loss)
        train_epoch_nap.append(tr_nap)
        val_epoch_nap.append(val_nap)

        metrics_pharma.plot_loss(
            train_epoch_loss, val_epoch_loss, save_dir=args.save, num_epoch=args.epochs
        )
        metrics_pharma.plot_nap(
            train_epoch_nap, val_epoch_nap, save_dir=args.save, num_epoch=args.epochs
        )

        logging.info(
            "Train:Loss {}, ACC {}, AUC {}, F-Measure {}, NAP {}".format(
                epoch_loss, tr_acc, tr_auc, tr_f, tr_nap
            )
        )
        logging.info(
            "Valid:Loss {}, ACC {}, AUC {}, F-Measure {}, NAP {}".format(
                val_loss, val_acc, val_auc, val_f, val_nap
            )
        )

        model.print_params(epoch=epoch)

        save_ckpt(
            model,
            optimizer,
            train_epoch_loss,
            val_epoch_loss,
            train_epoch_nap,
            val_epoch_nap,
            epoch,
            args.model_save_path,
            sub_dir,
            name_post="Last_model",
        )

        if tr_nap > results["highest_train"]:

            results["highest_train"] = tr_nap


        if val_loss < results["Lowest_valid_loss"]:
            results["highest_valid"] = val_nap
            results["epoch"] = epoch
            results["Lowest_valid_loss"] = val_loss

            save_ckpt(
                    model,
                    optimizer,
                    train_epoch_loss,
                    val_epoch_loss,
                    train_epoch_nap,
                    val_epoch_nap,
                    epoch,
                    args.model_save_path,
                    sub_dir,
                    name_post="valid_best",
                )

        if val_loss >= past_loss:
            loss_track += 1
        else:
            loss_track = 0

        past_loss = val_loss
        if loss_track >= 15:
            logging.info("Early exit due to overfitting")
            end_time = time.time()
            total_time = end_time - start_time
            logging.info("Best model in epoch: {}".format(results["epoch"]))
            logging.info(
                "Total time: {}".format(
                    time.strftime("%H:%M:%S", time.gmtime(total_time))
                )
            )
            sys.exit()

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Best model in epoch: {}".format(results["epoch"]))
    logging.info(
        "Total time: {}".format(time.strftime("%H:%M:%S", time.gmtime(total_time)))
    )


if __name__ == "__main__":
    cls_criterion = torch.nn.BCELoss()
    reg_criterion = torch.nn.MSELoss()
    main()
