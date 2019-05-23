#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from model import GraphMDANet
from utils import get_logger
from utils import data_loader
from utils import multi_data_loader

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="amazon")
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=1.0)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- " "not show training progress.", type=bool, default=True)
# parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
#                     type=str, default="gmdan")
# The experimental setting of using 5000 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=5000)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=5)
parser.add_argument("-bs", "--batch_size", help="Batch size during training", type=int, default=30)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")

parser.add_argument("-dr", "--dropout", help='Dropout rate (1 - keep probability).',  type=float, default=0.1)
parser.add_argument("-a",  "--alpha", help='Alpha for the leaky_relu.', type=float, default=0.005)
parser.add_argument("-b", "--margin", help='Positive to negative triplet distance margin.', type=float, default=1)
parser.add_argument("-hd",  "--nb_heads", help='Number of head attentions.', type=int, default=5)
parser.add_argument("-k",  "--k", help='k-Nearest Neighborhood for graphs.', type=int, default=5)

parser.add_argument("-mlr",  "--meta_lr", help='update learning rate for inner loop.', type=float, default=0.05)
parser.add_argument("-lr", "--update_lr", help='meta learning rate for outer loop.', type=float, default=0.05)
parser.add_argument("-up", "--update_step", help='a fixed update step for inner loop', type=int, default=4)


# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = {"hidden_layers": [1000, 500, 100], "gat_hidden_layers": [20], "num_classes": 2,
           "num_epochs": args.epoch, "num_trains": int(2000 * args.frac), "batch_size": args.batch_size, "dropout": args.dropout, "alpha": args.alpha,
           "nheads": args.nb_heads, "margin": args.margin, "k": args.k,  "mu": args.mu,
           "update_lr": args.update_lr, "meta_lr": args.meta_lr, "update_step": args.update_step,
           "mode": args.mode, "gamma": 1.0, "lambda": 0.0001, "verbose": args.verbose}



def load_data():
    logger = get_logger(args.name)

    # Set random number seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Loading the randomly partition the amazon data set.
    time_start = time.time()

    amazon = np.load("./amazon.npz")

    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :args.dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    time_end = time.time()
    logger.info("Time used to process the Amazon data set = {} seconds.".format(time_end - time_start))
    logger.info("Number of training instances = {}, number of features = {}."
                 .format(amazon_xx.shape[0], amazon_xx.shape[1]))
    logger.info("Number of nonzero elements = {}".format(amazon_xx.nnz))
    logger.info("amazon_xx shape = {}.".format(amazon_xx.shape))
    logger.info("amazon_yy shape = {}.".format(amazon_yy.shape))
    # Partition the data into four categories and for each category partition the data set into training and test set.
    configs["data_name"] = ["books", "dvd", "electronics", "kitchen"]
    configs["num_data_sets"] = len(configs["data_name"])
    configs["num_domains"] = configs["num_data_sets"] - 1
    data_insts, data_labels, num_insts = [], [], []
    for i in range(configs["num_data_sets"]):
        print ("----")

        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        print (np.sum(data_insts[i], 1).shape)
        logger.info("Length of the {} data set label list = {}, label values = {}, label balance = {}".format(
            configs["data_name"][i],
            amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :].shape[0],
            np.unique(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :]),
            np.sum(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :]) # how many positives
        ))
        num_insts.append(amazon_offset[i+1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
        # print (len(data_insts)) # 1, 2, 3, 4 equals the number of domain
        # print (data_insts[0].shape)  # each element of data_insts store  (6465, 5000) source domain
        # print (len(data_labels)) # 1, 2, 3, 4 store labels
        # print (data_labels[0].shape)
    logger.info("Data sets: {}".format(configs["data_name"]))
    logger.info("Number of total instances in the data sets: {}".format(num_insts))
    # Partition the data set into training and test parts, following the convention in the ICML-2012 paper, use a fixed
    # amount of instances as training and the rest as test.
    configs["input_dim"] = amazon_xx.shape[1]

    return data_insts, data_labels, logger


def train_batch(target_insts, xs, ys, model, optimizer): # inner loop
    batch_size = configs["batch_size"]
    num_domains = configs["num_domains"]

    for j in range(num_domains+1):
        xs[j] = torch.tensor(xs[j], requires_grad=False).to(device) # turn xs to tensor
        ys[j] = torch.tensor(ys[j], requires_grad=False).to(device) # turn ys to tensor

    ridx = np.random.choice(target_insts.shape[0], batch_size)
    tinputs = target_insts[ridx, :]
    tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
    optimizer.zero_grad() # zero_grad is what ?  Sets gradients of all model parameters to zero. why in each epoch, grad should be set to zero?
    losses, preds = model(xs, tinputs, ys) # the input is the current domain "tinputs" as well as the "xs"
    return losses, preds


def  train(data_insts, data_labels, logger):
        # The confusion matrix stores the prediction accuracy between the source and the target tasks. The row index the source
        # task and the column index the target task.
        results = {}
        logger.info("Training fraction = {}, number of actual training data instances = {}".format(args.frac, configs["num_trains"]))
        logger.info("-" * 100)

        batch_size = configs["batch_size"]
        num_trains = configs["num_trains"]
        num_epochs = configs["num_epochs"]
        num_domains = configs["num_domains"]
        num_data_sets = configs["num_data_sets"]

        meta_lr = configs["meta_lr"]
        mu = configs["mu"]
        gamma = configs["gamma"]
        mode = configs["mode"]
        lamda = configs["lambda"]

        logger.info("Training with domain adaptation using PyTorch madnNet: ")
        logger.info("Hyperparameter setting = {}.".format(configs))
        error_dicts = {}
        # print (num_domains) 3 -- number of source domains
        for i in range(num_data_sets): # for each domain, it trains the following model
            configs["test_task"] = configs["data_name"][i]
            print ("test task is: ", configs["test_task"])
            for j in range(num_data_sets):
                if i == j: continue
                configs["val_task"] = configs["data_name"][j]
                val_task_id = j
                print ("simulated source domain is :",  configs["val_task"])
                # Build source instances.
                source_insts = []
                source_labels = []
                # print (i)
                for j in range(num_data_sets): # add every dataset except the current source here
                    source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                    source_labels.append(data_labels[j][:num_trains, :].ravel().astype(np.int64))


                # Build target instances.
                target_idx = i # the current domain
                target_insts = data_insts[i][:num_trains, :].todense().astype(np.float32)
                target_labels = data_labels[i][:num_trains, :].ravel().astype(np.int64)

                # Train DannNet.
                model =  GraphMDANet(configs, device).to(device)
                optimizer = optim.Adadelta(model.parameters(), lr=meta_lr) # why Adadelta here ?
                model.train() # seems train is function by PyTorch

                # Training phase.
                time_start = time.time()
                for t in range(num_epochs): # 15 epoch by default
                    running_loss = 0.0
                    train_loader = multi_data_loader(source_insts, source_labels, batch_size) # containing instances and labels from multiple sources

                    for xs, ys in train_loader: # for each source-target pair
                        loss, preds = train_batch(target_insts, xs, ys, model, optimizer)

                        # print (loss.size())
                        running_loss += loss.item()
                        pred_acc = torch.sum(preds == ys[val_task_id]).item() / float(len(preds))
                        loss.backward()
                        optimizer.step()
                    logger.info("Iteration {}, loss = {}, pred_acc = {}".format(t, running_loss, pred_acc))
                    time_end = time.time()

                # Test on other domains.
                model.eval()
                val_target_insts = data_insts[i][num_trains:, :].todense().astype(np.float32)
                val_target_labels = data_labels[i][num_trains:, :].ravel().astype(np.int64)
                val_target_insts = torch.tensor(val_target_insts, requires_grad=False).to(device)
                val_target_labels = torch.tensor(val_target_labels)

                # generate source test data
                val_source_insts = []
                val_source_labels = []
                for j in range(num_data_sets):
                    if j != i:
                        val_source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                        val_source_labels.append(data_labels[i][:num_trains, :].ravel().astype(np.int64))
                for j in range(num_domains): # turn source to tensor
                    val_source_insts[j] = torch.tensor(val_source_insts[j], requires_grad=False).to(device)
                preds_labels = torch.max(model.inference(val_source_insts, val_target_insts, val_source_labels), 1)[1].cpu().data.squeeze_()

                val_target_labels = val_target_labels[:preds_labels.size()[0]]
                pred_acc = torch.sum(preds_labels == val_target_labels).item() / float(len(preds_labels))
                error_dicts[configs["data_name"][i]] = preds_labels.numpy() != val_target_labels.numpy()

                logger.info("Prediction accuracy on {} = {} ".
                            format(configs["data_name"][i], pred_acc))
                print ("-----")
                results[configs["data_name"][i]] = pred_acc
            logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
            logger.info(results)
            pickle.dump(error_dicts, open("{}-{}-{}-{}.pkl".format(args.name, args.frac, args.model, args.mode), "wb"))
            logger.info("*" * 100)



def main():
    print ("dataset: ", args.name)
    data_insts, data_labels, logger = load_data()
    # configs["input_dim"] = input_dim
    train(data_insts, data_labels, logger)

if __name__ == "__main__":
    main()
