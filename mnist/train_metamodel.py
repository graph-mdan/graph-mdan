import torch
import torchvision
import numpy as np
import utils
import time
import argparse
import pickle
import torch.optim as optim
from utils import get_logger
from dataset import MNIST_Dataset,get_dataloaders,data_transforms, split_name
from ournet_advance import OurNet
import torch.nn.functional as F
from utils import multi_data_loader
from metamodel import Mdan




def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    results = {}
    data_insts, data_labels, num_insts = [], [], []
    # mnist_m = MNIST_Dataset('MNIST_M',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
    # mnist = MNIST_Dataset('MNIST',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
    # synthdigits = MNIST_Dataset('SYNTHDIGITS',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
    
    num_data_sets = 4
    dataset_names = ['MNIST_M','SVHN','MNIST','SYNTHDIGITS']
    for i in range(num_data_sets):
        dataset = MNIST_Dataset(dataset_names[i],is_train = True,
            max_data_num =20000,transform=data_transforms[split_name[True]])
        
        tmp_data ,tmp_label = [],[]
        for j in range(len(dataset)):
            tmp_data.append(dataset[j][0])
            tmp_label.append(dataset[j][1])
        
        tmp_data =  torch.stack(tmp_data)
        tmp_label = np.array(tmp_label)
        
        data_insts.append(tmp_data)
        
        data_labels.append(tmp_label)


   
        
       
        
    
    if args.model=="mdan":
        configs = {"num_classes": 10,
               "num_epochs":5, "batch_size": 5, "lr": 1e-1, "mu": 10, "num_domains":
                   3,  "gamma": 10.0,  "lambda": 0.01, 'margin':0.1, 'dropout':0,'k':2,'alpha':0.2, 'device':device, 
                 "update_lr": 0.05, "meta_lr": 0.05, "update_step": 4 }
        configs["data_name"] = ['MNIST_M','SVHN','MNIST','SYNTHDIGITS']

        num_epochs = configs["num_epochs"]
        batch_size = configs["batch_size"]
        num_domains = configs["num_domains"]
        lr = configs["lr"]
        mu = configs["mu"]
        gamma = configs["gamma"]
        lamda = configs["lambda"]
        
        logger.info("Training with domain adaptation using PyTorch madnNet: ")
        logger.info("Hyperparameter setting = {}.".format(configs))
    
        
        error_dicts = {}
        target_data_insts, target_data_labels = [],[]
        for i in range(num_data_sets):
            # Build source instances.
            configs["test_task"] = configs["data_name"][i]

            source_insts = []
            source_labels = []
            infer_source_insts =[]
            infer_source_labels =[]
            for j in range(num_data_sets):
                if j != i:
                    configs["val_task"] = configs["data_name"][j]
                    val_task_id = j 
                    source_insts.append(data_insts[j][:,:,:,:].numpy().astype(np.float32))
                    source_labels.append(data_labels[j][:].ravel().astype(np.int64))
                    
            
            target_idx = i
            target_dataset = MNIST_Dataset(dataset_names[i],is_train = False,
                      max_data_num =20000,transform=data_transforms[split_name[False]])
            tmp_data ,tmp_label = [],[]
            for k in range(len(target_dataset)):
                tmp_data.append(target_dataset[k][0])
                tmp_label.append(target_dataset[k][1])
        
            tmp_data =  torch.stack(tmp_data)
            tmp_label = np.array(tmp_label)
        
            target_data_insts.append(tmp_data)
            
            target_data_labels.append(tmp_label)

            target_insts = target_data_insts[i][:,:,:,:]
            target_labels = target_data_labels[i][:].ravel().astype(np.int64)
            
           
            #model = OurNet(configs).to(device)
            print('all good until now')
            model = Mdan(configs).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
            model.train()
            # Training.
            
            time_start = time.time()
            for t in range(num_epochs):
                running_loss = 0.0
                train_loader = multi_data_loader(source_insts, source_labels, batch_size)
                for xs, ys in train_loader:
                    
                    slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                    tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                    for j in range(num_domains):
                        xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                        
                        ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                        
                    ridx = np.random.choice(target_insts.shape[0], batch_size)
                    tinputs = target_insts[ridx, :]
                    tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                    
                    optimizer.zero_grad()
                    logprobs, sdomains, tdomains= model(xs, tinputs, ys)
                    #logprobs, sdomains, tdomains= model(xs, tinputs)
                    #print('tinputsshape', tinputs.shape)
                    # Compute prediction accuracy on multiple training sources.
                    
                    losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)])
                    domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                            F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                    
                    if mode == "maxmin":
                        loss = torch.max(losses) + mu * torch.min(domain_losses)
                    elif mode == "dynamic":
                        loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                        
                    else:
                        raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                logger.info("Iteration {}, loss = {}".format(t, torch.max(losses).item()))
                logger.info("Iteration {}, loss = {}".format(t, loss.item()))
                logger.info("Iteration {}, loss = {}".format(t, running_loss))
               
            time_end = time.time()
            # Test on other domains.
            model.eval()
            # target_insts = data_insts[i][:].astype(np.float32)
            # target_insts = torch.tensor(target_insts, requires_grad=False).to(device)
            
            # target_labels = data_labels[i][:].ravel().astype(np.float32)
            # target_labels = torch.tensor(target_labels, dtype=torch.long)
            
            target_labels = torch.tensor(target_labels, requires_grad=False, dtype= torch.long).cpu().data.squeeze_() # numpy 2 tensor
            model = model.cpu()
            preds_labels = torch.max(model.inference(target_insts), 1)[1].cpu().data.squeeze_()
            preds_labels = torch.tensor(preds_labels, requires_grad=False, dtype= torch.long) # numpy 2 tensor
            pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
            print(preds_labels.shape)
            print(target_labels.shape)
            print(torch.sum(preds_labels == target_labels).item())
            print(target_insts.size(0))
            
            #pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
            
            
            logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".
                        format(dataset_names[i], pred_acc, time_end - time_start))
            results[dataset_names[i]] = pred_acc
        logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
        logger.info(results)
        
        logger.info("*" * 100)
    else:
        raise ValueError("No support for the following model: {}.".format(args.model))



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="amazon")
    parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                        type=float, default=1.0)
    parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
    parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                                "not show training progress.", type=bool, default=True)
    parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                        type=str, default="mdan")
    parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                        type=float, default=1e-2)
    parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=40)
    parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=60)
    parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(args.name)
    # Set random number seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)

