import torch as pt
import torch.nn as nn
import os
import numpy as np
import random

from collections import defaultdict

import pickle
import wandb
import yaml

from load_data import load_all_data, load_data
from load_model import load_model
from optimizer import init_optimizer
from compute_NCmetrics import log_results, log_training

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


seeds = [3141, 1998, 2025, 1234, 2718, 1996, 1357, 2468, 9876, 1000]


def accuracy(out, yb):
    '''
    calculates the accuracy based on the predicted outputs and true labels y

    out: predicted output
    yb: true labels y
    '''
    preds = pt.argmax(out, axis=1)
    acc = (preds == yb).float().mean()
    
    return acc



@hydra.main(config_path="conf", config_name="config")
def main(config):

    # create foldr for figures and histories, if they do not already exist
    os.makedirs("histories", exist_ok = True)
    os.makedirs("figures", exist_ok = True)
    
    # Hyperparameters
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    # device = config['device'] 

    
    job_idx = HydraConfig.get().job.num  # Unique job number assigned by Hydra
    gpu_id = (job_idx % 6) # Cycles through 0,1,2,3,4,5 for your 6 GPUs
    # Set the GPU environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Job {job_idx} running on GPU {gpu_id}")
    # Explicitly select the GPU in PyTorch
    device = pt.device("cuda:0")  # Since we masked visibility to only one GPU

    
    log_run = config['log_run']
    log = config['log']
    log_every = config['log_every']
    
    num_runs = config['num_runs']
    assert(num_runs <= len(seeds)), f'Number of seeds not sufficient! #Num_runs={num_runs} > #seeds={len(seeds)}'
    
    print(f"Running {num_runs} experiments with the following configuration:")
    print(OmegaConf.to_yaml(config))

    for i in range(num_runs):
        seed = seeds[i]
        pt.manual_seed(seed)
        pt.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        lr, epochs_lr_decay, lr_decay = config['lr'], [int(n_epochs/3), int(2*n_epochs/3)], config['lr_decay']
        momentum = config['momentum']
        weight_decay = config['weight_decay'] #1e-4, 5e-4
        weight_decay_coupled = config['weight_decay_coupled']
        weight_decay_L1 = config['weight_decay_L1']
        lr_schedule = config['lr_schedule']
        model_name = config['model_name']
        pretrained_weights = config['pretrained']
        finetune = config['finetune']

        opt = config['opt_name']
        dataset_name = config['dataset_name']
        ood_dataset_name = config['ood_dataset_name']

        num_groups = config['num_groups'] # only relevant for GN


        # Prepare dataset
        print('pretrained_weights:', pretrained_weights)
        if pretrained_weights != 'none':
            img_sz = 224 # probably loading weights pretrained on ImageNet
        else:
            img_sz = 32
        train_loader, test_loader, ood_loader = load_all_data(dataset_name, ood_dataset_name, batch_size, img_sz)
        
        model, last_layer_width = load_model(model_name, dataset_name, device, pretrained_weights, finetune, num_groups)



        total_training_steps = len(train_loader)*n_epochs

        class Features:
            pass


        def hook(self, input, output):
            Features.value = input[0].clone()


        # register hook that saves last-layer input into features
        classifier = model.fc
        hook_handle = classifier.register_forward_hook(hook)



        # Loss and optimizer
        if config['loss_func'] == 'CE':
            criterion = nn.CrossEntropyLoss()
        else:
            ValueError(f'Code is not implemented with {config["loss_func"]} as loss')


        # Initialize Optimizer and LR scheduler
        optimizer, lr_scheduler = init_optimizer(opt, lr, model, model_name, lr_schedule, momentum=momentum, weight_decay=weight_decay,
                                                 total_training_steps=total_training_steps, warmup_length=0.05, epochs_lr_decay=epochs_lr_decay, 
                                                 lr_decay=lr_decay, weight_decay_coupled=weight_decay_coupled,
                                                weight_decay_L1=weight_decay_L1)



        if log_run:

            wandb.init(
                    project=f"{config['project_name']}",
                    # track hyperparameters and run metadata
                    config={
                    "experiment": config['experiment_name'],
                    "optimizer": opt,
                    "model": model_name,
                    "epochs": n_epochs,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "weight_decay_coupled": weight_decay_coupled,
                    "weight_decay_L1": weight_decay_L1,
                    "lr": lr,
                    "lr_schedule": lr_schedule,
                    "dataset": dataset_name,
                    "ood_dataset": ood_dataset_name,
                    "batchsize": batch_size,
                    "pretrained": pretrained_weights,
                    "finetune": finetune,
                    "num_groups": num_groups,
                    "seed": seed
                    }
                )


        # Train the model
        total_step = len(train_loader)
        log_line = lambda epoch, i: f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_step}]"
        history = defaultdict(list)

        for epoch in range(n_epochs):
            model.train()

            # Accumulators
            running_loss = 0.0
            running_correct = 0
            total_samples = 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()

                if model_name == 'ViT_B_16' or model_name == 'Vit_custom': # gradient clipping
                    pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Only for Cosine decay
                if lr_schedule == 'cosine_decay':
                    lr_scheduler.step()
                    
                if log_run and ((epoch % log_every) == 0 or (epoch == n_epochs-1)):

                    # ---- Accumulate training loss ----
                    # Multiply by the batch size so we have the total loss for these samples
                    running_loss += loss.item() * images.size(0)

                    # ---- Accumulate training accuracy ----
                    # Use your accuracy() function or do an equivalent inline
                    batch_acc = accuracy(logits, labels)  
                    running_correct += batch_acc.item() * images.size(0)

                    # Track how many total samples we've seen
                    total_samples += images.size(0)

                    
                if lr_scheduler == None:
                    learning_rate = lr
                else:
                    learning_rate = lr_scheduler.get_last_lr()[0]

                if (i + 1) % 150 == 0:
                    print(f"{log_line(epoch, i)}, Loss: {loss.item():.4f}, LR: {learning_rate}")


            if log_run and ((epoch % log_every) == 0 or (epoch == n_epochs-1)):
                # Compute the average training loss & accuracy over the entire dataset
                epoch_train_loss = running_loss / total_samples
                epoch_train_acc = running_correct / total_samples


            if lr_schedule == 'step_decay':
                # Only for models other than ViT
                lr_scheduler.step()

            if log_run:
                if (epoch % log_every) == 0 or (epoch == n_epochs-1):
                    if log == 'NC_metrics':

                        log_results(model,  # model state should be safe if you're not modifying it
                                    train_loader,
                                    test_loader,
                                    ood_loader,
                                    device,
                                    last_layer_width,
                                    criterion, 
                                    accuracy, 
                                    Features,
                                    epoch,
                                    epoch_train_loss,
                                    epoch_train_acc,
                                    learning_rate,
                                    history,
                                    dataset_name
                                )

                    elif log == 'training':

                        log_training(model,  # model state should be safe if you're not modifying it
                            test_loader,
                            device,
                            criterion, 
                            accuracy, 
                            epoch,
                            epoch_train_loss,
                            epoch_train_acc,
                            learning_rate
                        )
                    else:
                        raise ValueError(f'Unknown log setting {log}')



        if log_run:

            wandb.finish()
            wandb.join()


            # Save results
            if log == 'NC_metrics':

                with open(f'histories/history_{config["project_name"]}_{config["experiment_name"]}_{dataset_name}_{model_name}_{opt}_lr={lr}_wd={weight_decay}.pickle', 'wb') as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    

    pt.cuda.empty_cache()

    
    main()
    # load in YAML configuration

#     config = {}
#     base_config_path = 'conf/config.yaml'
#     with open(base_config_path, 'r') as file:
#         config.update(yaml.safe_load(file))

#     # start training with name and config 
#     main(config)
