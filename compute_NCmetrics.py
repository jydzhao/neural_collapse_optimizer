from neural_collapse.accumulate import (CovarAccumulator, DecAccumulator,
                                        MeanAccumulator, VarNormAccumulator)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (clf_ncc_agreement, covariance_ratio,
                                     orthogonality_deviation,
                                     self_duality_error, similarities,
                                     simplex_etf_error, variability_cdnv,
                                     structure_error, 
                                     norm_cov, self_duality2_error,
                                     equiangular_M_error,equiangular_W_error,
                                     )

import torch as pt

from load_data import load_data
import wandb

def log_ood_training(model, ood_loader, device, criterion, accuracy,
                epoch,
                epoch_train_loss,
                epoch_train_acc,
                lr):
    """Logs all results to WandB
    """

    results = compute_test_loss(model, ood_loader, device, criterion, accuracy)

    results["epoch"] = epoch
    results["train_loss"] = epoch_train_loss
    results["train_accuracy"] = epoch_train_acc
    results["lrs"] = lr

    wandb.log(results, commit=True)

    return results

def log_training(model, test_loader, device, criterion, accuracy,
                epoch,
                epoch_train_loss,
                epoch_train_acc,
                lr):
    """Logs all results to WandB
    """

    results = compute_test_loss(model, test_loader, device, criterion, accuracy)

    results["epoch"] = epoch
    results["train_loss"] = epoch_train_loss
    results["train_accuracy"] = epoch_train_acc
    results["lrs"] = lr

    wandb.log(results, commit=True)

    return results


def log_results(model, train_loader, test_loader, ood_loader, device, width, criterion, accuracy, Features, 
                epoch,
                epoch_train_loss,
                epoch_train_acc,
                lr, history,
                train_dataset_name):
    """Logs all results to WandB
    """

    results = compute_metrics(model, train_loader, test_loader, ood_loader, device, width, criterion, accuracy, Features, train_dataset_name)

    results["epoch"] = epoch
    results["train_loss"] = epoch_train_loss
    results["train_accuracy"] = epoch_train_acc
    results["lrs"] = lr

    wandb.log(results, commit=True)

    history['epoch'].append(epoch)
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(results["val_loss"])
    history['val_acc'].append(results["val_accuracy"])
    history['NC1'].append(results["NC1"])
    history['NC2n'].append(results["NC2n"])
    history['NC2a'].append(results["NC2a"])
    history['NC2'].append(results["NC2"])
    history['NC2W'].append(results["NC2W"])
    history['NC2Wn'].append(results["NC2W"])
    history['NC2Wa'].append(results["NC2Wa"])
    history['NC2M'].append(results["NC2M"])
    history['NC3'].append(results["NC3"])
    history['NC4'].append(results["NC4"])
    history['NC0'].append(results["norm(last_layer_rowsum)"])
    history['lr'].append(lr)

    return results, history

def compute_test_loss(model, test_loader, device, criterion, accuracy):
    """Compute test loss and accuracy
       This function is intended to run asynchronously.
    """

    print("Computing test loss...")
    
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_samples = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        val_loss += loss.item() * images.size(0)
        val_correct += accuracy(outputs, labels).item() * images.size(0)
        val_samples += images.size(0)
    
    val_loss /= val_samples
    val_accuracy = val_correct / val_samples

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }

    return results

    

def compute_metrics(model, train_loader, test_loader, ood_loader, device, width, criterion, accuracy, Features, train_dataset_name):
    """Computes all your metrics and returns a dictionary of results.
       This function is intended to run asynchronously.
    """

    print("Computing metrics...")
    
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_samples = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        val_loss += loss.item() * images.size(0)
        val_correct += accuracy(outputs, labels).item() * images.size(0)
        val_samples += images.size(0)
    
    val_loss /= val_samples
    val_accuracy = val_correct / val_samples
    
#     ood_loss = 0.0
#     ood_correct = 0
#     ood_samples = 0
    
#     for images, labels in ood_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         ood_loss += loss.item() * images.size(0)
#         ood_correct += accuracy(outputs, labels).item() * images.size(0)
#         ood_samples += images.size(0)
    
#     ood_loss /= ood_samples
#     ood_accuracy = ood_correct / ood_samples

    # with pt.no_grad():
    weights = model.fc.weight.clone()

    means, mG, var_norms, covar_within, dec_accum, mG_ood = compute_NCmetrics(model, train_loader, test_loader, ood_loader, device, width, weights,
                                                                              criterion, accuracy, Features, train_dataset_name)

    # M_spectrum = pt.linalg.svdvals(means - mG).detach().cpu().numpy().tolist()
    # W_spectrum = pt.linalg.svdvals(weights).detach().cpu().numpy().tolist()

    if mG_ood is None:
        NC5 = None
    else:
        NC5 = orthogonality_deviation(means, mG_ood)
    
    results = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
#         "ood_loss": ood_loss,
#         "ood_accuracy": ood_accuracy,
        # "M_spectrum": M_spectrum,
        # "W_spectrum": W_spectrum,
        
        "NC1": covariance_ratio(covar_within, means, mG),
        "NC2n": norm_cov(means, mG),
        "NC2a": equiangular_M_error(means, mG),
        "NC2": simplex_etf_error(means, mG),
        "NC2W": structure_error(weights, weights),
        "NC2Wn": norm_cov(weights),
        "NC2Wa": equiangular_W_error(weights),
        "NC2M": self_duality_error(weights, means, mG),
        "NC3": self_duality2_error(weights, means, mG),
        "NC4": clf_ncc_agreement(dec_accum),
        
        "nc1_svd": covariance_ratio(covar_within, means, mG, "svd"),
        "nc1_quot": covariance_ratio(covar_within, means, mG, "quotient"),
        "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
        "nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
        "nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1],
        "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
        "nc5_ood_dev": NC5,
        "norm(last_layer_rowsum)": pt.linalg.norm(pt.sum(weights,axis=0))
    }
    # pt.cuda.synchronize()

    
    print("Returning metrics")

    return results
    

def compute_NCmetrics(model, train_loader, test_loader, ood_loader, device, width, weights, criterion, accuracy, Features, train_dataset):
    
    # --- Example: Collect training features ---
    all_train_features, all_train_labels = [], []
    model.eval()
    
    if train_dataset == 'cifar100':
        n_classes = 100
    else:
        n_classes = 10
    print(f'num_classes = {n_classes}')
    
    with pt.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            _ = model(images)  # Hook stores features in Features.value
            all_train_features.append(Features.value.clone().detach().cpu())
            all_train_labels.append(labels.cpu())
            
        all_train_features = pt.cat(all_train_features, dim=0)
        all_train_labels = pt.cat(all_train_labels, dim=0)
        
        mean_accum = MeanAccumulator(n_classes, width, device)
        mean_accum.accumulate(all_train_features.to(device), all_train_labels.to(device))
        means, mG = mean_accum.compute()
        var_norms_accum = VarNormAccumulator(n_classes, width, device, M=means)
        covar_accum = CovarAccumulator(n_classes, width, device, M=means)
        var_norms_accum.accumulate(all_train_features.to(device), all_train_labels.to(device), means)
        covar_accum.accumulate(all_train_features.to(device), all_train_labels.to(device), means)
        var_norms, _ = var_norms_accum.compute()
        covar_within = covar_accum.compute()
    
        
        all_test_features, all_test_labels = [], []
        for images, labels in test_loader:
            images = images.to(device)
            _ = model(images)  # Hook stores features in Features.value
            all_test_features.append(Features.value.clone().detach().cpu())
            all_test_labels.append(labels.cpu())
        all_test_features = pt.cat(all_test_features, dim=0)
        all_test_labels = pt.cat(all_test_labels, dim=0)
    
        dec_accum = DecAccumulator(n_classes, width, device, M=means, W=weights)
        dec_accum.create_index(means)  # optionally use FAISS index for NCC
        
        # mean embeddings (only) necessary again if not using FAISS index
        if dec_accum.index is None:
            dec_accum.accumulate(all_test_features.to(device), all_test_labels.to(device), weights, means)
        else:
            dec_accum.accumulate(all_test_features.to(device), all_test_labels.to(device), means)


        if train_dataset != 'cifar100':
            
            all_ood_features, all_ood_labels = [], []
            for images, labels in ood_loader:
                images = images.to(device)
                _ = model(images)  # Hook stores features in Features.value
                all_ood_features.append(Features.value.clone().detach().cpu())
                all_ood_labels.append(labels.cpu())
            all_ood_features = pt.cat(all_ood_features, dim=0)
            all_ood_labels = pt.cat(all_ood_labels, dim=0)
            
            ood_mean_accum = MeanAccumulator(10, width, device)
            ood_mean_accum.accumulate(all_ood_features, all_ood_labels)
            _, mG_ood = ood_mean_accum.compute()

        else:
            print('no OOD dataset provided, set mG_ood to None')
            mG_ood = None

    return means, mG, var_norms, covar_within, dec_accum, mG_ood

    