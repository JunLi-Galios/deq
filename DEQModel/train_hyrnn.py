# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
import prefix_dataset
import models.hyrnn.model as model
from tqdm import tqdm

from data_utils import get_lm_corpus
from models.transformers.deq_transformer import DEQTransformerLM
from modules import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel


parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence of hyrnn Model')

parser.add_argument("--data_dir", type=str, help="", default="./data")
parser.add_argument("--data_class", type=int, help="", default=10)
parser.add_argument("--num_epochs", type=int, help="", default=100)
parser.add_argument("--log_dir", type=str, help="", default="logdir")
parser.add_argument("--batch_size", type=int, help="", default=64)

parser.add_argument("--embedding_dim", type=int, help="", default=5)
parser.add_argument("--hidden_dim", type=int, help="", default=5)
parser.add_argument("--project_dim", type=int, help="", default=5)
parser.add_argument("--use_distance_as_feature", action="store_true", default="True")


parser.add_argument("--num_layers", type=int, help="", default=1)
parser.add_argument("--verbose", type=bool, help="", default=True)
parser.add_argument(
    "--cell_type", choices=("hyp_gru", "eucl_rnn", "eucl_gru"), default="eucl_gru"
)
parser.add_argument("--decision_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--embedding_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--sgd", action='store_true')
parser.add_argument("--adam_betas", type=str, default="0.9,0.999")
parser.add_argument("--wd", type=float, default=0.)
parser.add_argument("--c", type=float, default=1.)
parser.add_argument("--j", type=int, default=1)
parser.add_argument("--save_epoch", type=int, default=10)

args = parser.parse_args()
if not os.path.exists("./logs"):
    os.mkdir("./logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = args.data_dir
logdir = os.path.join("./logs", args.log_dir)
logging = create_exp_dir(args.log_dir,
    scripts_to_save=['train_hyrnn.py', 'models/hyrnn/model.py'], debug=False)

n_epochs = args.num_epochs
num = args.data_class
batch_size = args.batch_size
adam_betas = args.adam_betas.split(",")

dataset_train = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="train", download=True
)

dataset_test = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="test", download=True
)

loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn,
    shuffle=True, num_workers=args.j,
)

loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn
)

###############################################################################
# Build the model
###############################################################################
model = model.RNNBase(
    dataset_train.vocab_size,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    project_dim=args.project_dim,
    cell_type=args.cell_type,
    device=device,
    num_layers=args.num_layers,
    use_distance_as_feature=args.use_distance_as_feature,
    num_classes=2,
    c=args.c
).to(device)

criterion = nn.CrossEntropyLoss()
if not args.sgd:
    optimizer = geoopt.optim.RiemannianAdam(
        model.parameters(),
        lr=args.lr,
        betas=(float(adam_betas[0]), float(adam_betas[1])),
        stabilize=10,
        weight_decay=args.wd
    )
else:
    optimizer = geoopt.optim.RiemannianSGD(
        model.parameters(), args.lr, stabilize=10,
        weight_decay=args.wd)

###############################################################################
# Training code
###############################################################################

def train(net, optim, criterion, train_loader, test_loader):
    print("\nStart Train len :", len(train_loader.dataset))        
    net.train()

    for epoch in range(args.num_epochs):
        for i, (input_) in enumerate(tqdm(train_loader)):
#             print('input_', input_)
#             input_ = input_.to(device)
            target_ = input_[-1].to(device)

            out = net(input_)
            loss = criterion(out, target_)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i % 50) == 0:
                logging("train, epoch={}, loss={}".format(epoch, loss.item()))
        acc = get_acc(net, train_loader)
        logging("train, epoch={}, acc={}".format(epoch, acc))
        if test_loader is not None:
            self.valid(epoch, val_loader)

def get_acc(net, loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for input_, target_ in loader:
            input_ = input_.to(device)
            out = net(input_)
            out = F.softmax(out, dim=1)

            _, idx = out.max(dim=1)
            correct += (target_ == idx).sum().item()
    net.train()
    return correct / len(loader.dataset)

def valid(epoch, val_loader):
    acc = get_acc(val_loader)
    logging("test, ", epoch=epoch, acc=acc)

    if acc > best_acc:
        best_acc = acc
    if acc > best_acc or (epoch + 1)%save_epoch==0:
        save(epoch, "epoch[%05d]_acc[%.4f]" % (
            epoch, acc))
        
def save(epoch, filename="train"):
        """Save current epoch model
        Save Elements:
            model_type : model name
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score
        Parameters:
            epoch : current epoch
            filename : model save file name
        """

        torch.save({"model_type": self.model_name,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

# Loop over epochs.
best_acc = 0

train(model, optimizer, criterion, loader_train, loader_test)
