"""
    Some handy functions for pytroch model training ...
"""
import torch


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(
                                device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(optimizer,
                  learning_rate: float = None,
                  momentum: float = None,
                  weight_decay: float = None,
                  alpha: float = None,
                  network: torch.nn.Module = None):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=learning_rate,
                                        alpha=alpha,
                                        momentum=momentum)
    return optimizer
