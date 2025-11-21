# src/utils.py

import torch
import numpy as np

from torch import nn


def initialize_weights(m, init_method="kaiming_uniform"):
    """
    Linear layer weight init helper
    init_method: "xavier_uniform", "xavier_normal",
                 "kaiming_uniform", "kaiming_normal"
    """
    if isinstance(m, nn.Linear):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(m.weight)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        if m.bias is not None:
            nn.init.zeros_(m.bias)


def inference(
    model,
    dataloader,
    device=None,
    disable_permutation=True,
    return_tensor=False,
    return_target=False,
    mc_dropout=False,
    T=20,
):
    """
    Common inference function

    Returns
    -------
    - mc_dropout=False: np.ndarray (N,)
    - mc_dropout=True: np.ndarray (N, 2)  # mean, std
    """
    model.eval()
    if mc_dropout:
        # Dropout active
        model.train()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input"].to(device) if device is not None else batch["input"]

            if mc_dropout:
                preds_T = []
                for _ in range(T):
                    preds = model(x, disable_permutation=True).squeeze()
                    preds_T.append(preds.unsqueeze(0))  
                preds_T = torch.cat(preds_T, dim=0)     
                mean_preds = preds_T.mean(dim=0)         
                std_preds = preds_T.std(dim=0)           
                all_preds.append(torch.stack([mean_preds, std_preds], dim=1)) 
            else:
                preds = model(x, disable_permutation=True).squeeze()
                all_preds.append(preds)

            if return_target:
                all_targets.append(batch["target"].to(device) if device is not None else batch["target"])

    all_preds = torch.cat(all_preds, dim=0)

    if return_target:
        all_targets = torch.cat(all_targets, dim=0)
        if return_tensor:
            return all_preds, all_targets
        else:
            return all_preds.cpu().numpy(), all_targets.cpu().numpy()

    return all_preds if return_tensor else all_preds.cpu().numpy()

