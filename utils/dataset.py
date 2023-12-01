import torch
from lib.datasets import get_dataset
from torch.utils.data import DataLoader, random_split


def load_dataset(config, num_clients):
    """Loads the and prepare the dataset for federated learning simulation, partitioning it among clients"""
    gpus = list(config.GPUS)
    dataset_type = get_dataset(config)
    train_set = dataset_type(config, is_train=0)        # TODO: check 1/0
    val_set = dataset_type(config, is_train=1)
    
    # Split training and validation set into num_clients partitions to simulate the individual 
    train_set = ds_partition(train_set, num_clients)
    # val_set = ds_partition(val_set, num_clients)
    # TODO: non dividiamo il valid test, facciamo una valid complessiva

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    
    for ds_train in train_set:
        train_loaders.append(DataLoader(
            ds_train, 
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus), 
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
            ))
    
    val_loader = DataLoader(
        val_set, 
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        )
    return train_loaders, val_loader


def ds_partition(set_, num_clients):
    """Split the dataset into num_clients partitions"""
    partition_size = len(set_) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] = lengths[-1]+len(set_)-partition_size*num_clients
    set_ = random_split(set_, lengths, torch.Generator().manual_seed(42))
    return set_
