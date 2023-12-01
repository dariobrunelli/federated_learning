import torch
from utils.train_fns import train_fn, validate


def train(net, trainloader, optimizer, epochs, device, config):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    best_nme = 100

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR)
        
    for epoch in range(1): # FIXME: numero di epoche? è 60 ma significherebbe 60 per cliente per giro
        train_fn(config, trainloader, net, criterion, optimizer, epoch, None)
        lr_scheduler.step()
        
    return net


def test(net, testloader, config):
    """Validate the network on the entire test set."""
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    net.eval()
    with torch.no_grad():
        loss, nme = validate(config, testloader, net, criterion)
    
    return loss, nme
    