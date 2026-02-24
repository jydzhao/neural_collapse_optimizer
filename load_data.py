from gaussian_data import *
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomCrop, RandomHorizontalFlip, RandAugment

# Prepare dataset
def load_all_data(dataset_name, ood_dataset_name, batch_size, img_sz):
    if dataset_name == 'gaussian_mixture':
        d=784
        K=10
        n=1000
        train_dataset, test_dataset = generate_gaussian_data(d, K, n)
    elif dataset_name == 'mnist':
        # MNIST dataset 
        transform = Compose([Resize(img_sz), ToTensor(), Normalize((0.1307,), (0.3081,))]) # normalization for MNIST
        train_dataset = MNIST("~/partial_neural_collapse/data", True, transform, download=True)
        test_dataset = MNIST("~/partial_neural_collapse/data", False, transform, download=True)
    elif dataset_name == 'fashion':
        # transform = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))]) # Old normalization from MNIST used for Fashion before
        transform = Compose([Resize(img_sz), ToTensor(), Normalize((0.2860,), (0.3530,))]) 
        train_dataset = FashionMNIST("~/partial_neural_collapse/data", True, transform, download=True)
        test_dataset = FashionMNIST("~/partial_neural_collapse/data", False, transform, download=True)
    elif dataset_name == 'cifar10':        
        # transform_train = Compose([RandomCrop(32, padding=4),
        #                              RandomHorizontalFlip(),
        #                              ToTensor(), 
        #                              Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 
        # transform_test = Compose([
        #                          ToTensor(), 
        #                          Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 
        transform_train = transforms.Compose(
                                    [transforms.RandomCrop(32, padding=4),
                                     Resize(img_sz),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # transforms.ToTensor(),
                                    # transforms.Resize((32, 32)),
                                    # transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 channel means
                                                            std=[0.2470, 0.2435, 0.2616]     # CIFAR-10 channel stds
                                                        )])
        transform_test = transforms.Compose(
                                    [Resize(img_sz),
                                     transforms.ToTensor(),
                                    # transforms.Resize((32, 32)),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 channel means
                                                            std=[0.2470, 0.2435, 0.2616]     # CIFAR-10 channel stds
                                                        )])


        # N = 2; M = 10;
        # transform_train.transforms.insert(0, RandAugment(N, M))
        
        train_dataset = CIFAR10("~/partial_neural_collapse/data", True, transform_train, download=True)
        test_dataset = CIFAR10("~/partial_neural_collapse/data", False, transform_test, download=True)
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose(
                                    [transforms.RandomCrop(32, padding=4),
                                     Resize(img_sz),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-10 channel means
                                                            std=[0.2675, 0.2565, 0.2761]     # CIFAR-10 channel stds
                                                        )])
        transform_test = transforms.Compose(
                                    [Resize(img_sz),
                                     transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-10 channel means
                                                            std=[0.2675, 0.2565, 0.2761]     # CIFAR-10 channel stds
                                                        )])
        train_dataset = CIFAR100("~/partial_neural_collapse/data", True, transform_train, download=True)
        test_dataset = CIFAR100("~/partial_neural_collapse/data", False, transform_test, download=True)
        
    else:
        raise ValueError(f'Unknown dataset_name {dataset_name}')
    
    # Prepare dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,  multiprocessing_context='fork')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context='fork')
    
    # OoD dataset for NC5
    assert (dataset_name != ood_dataset_name)
    
    if ood_dataset_name == 'mnist':
        transform = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))]) 
        ood_dataset = MNIST("~/partial_neural_collapse/data", False, transform, download=True)
    elif ood_dataset_name == 'fashion':
        transform = Compose([Resize(32), ToTensor(), Normalize((0.2860,), (0.3530,))]) 
        ood_dataset = FashionMNIST("~/partial_neural_collapse/data", False, transform, download=True)
    elif ood_dataset_name == 'svhn': # transformation needed for SVHN?? 
        transform = Compose([Resize(32),ToTensor()])
        ood_dataset = SVHN("~/partial_neural_collapse/data", 'test', transform, download=True)
    elif ood_dataset_name == 'cifar10':
        transform = transforms.Compose(
                                    [Resize(img_sz),
                                     transforms.ToTensor(),
                                    # transforms.Resize((32, 32)),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 channel means
                                                            std=[0.2470, 0.2435, 0.2616]     # CIFAR-10 channel stds
                                                        )])
        ood_dataset = CIFAR10("~/partial_neural_collapse/data", False, transform, download=True)
    elif ood_dataset_name == '':
        pass
    else:
        raise ValueError(f'Unknown ood_dataset_name {ood_dataset_name}')

    if ood_dataset_name != '':
        ood_loader = DataLoader(dataset=ood_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,  multiprocessing_context='fork')
    else:
        ood_loader = None
        
    print('Dataset loaded')
    
    return train_loader, test_loader, ood_loader

def load_data(dataset_name, batch_size, train=True):
    if dataset_name == 'gaussian_mixture':
        d=784
        K=10
        n=1000
        train_dataset, test_dataset = generate_gaussian_data(d, K, n)
    elif dataset_name == 'mnist':
        # MNIST dataset 
        transform = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))]) # normalization for MNIST
        dataset = MNIST("~/partial_neural_collapse/data", train, transform, download=True)
    elif dataset_name == 'fashion':
        # transform = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))]) # Old normalization from MNIST used for Fashion before
        transform = Compose([Resize(32), ToTensor(), Normalize((0.2860,), (0.3530,))]) 
        dataset = FashionMNIST("~/partial_neural_collapse/data", train, transform, download=True)
    elif dataset_name == 'cifar10':
        transform = Compose([ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]) 
        dataset = CIFAR10("~/partial_neural_collapse/data", train, transform, download=True)
    else:
        raise ValueError(f'Unknown dataset_name {dataset_name}')
    
    # Prepare dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context='fork')


    return data_loader
    