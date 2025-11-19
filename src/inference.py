# src/inference.py

import argparse
import json
import numpy as np
import pandas as pd      # ← 이 줄이 반드시 필요함!!!
import torch
from torch.utils.data import DataLoader

from .model import build_viscosity_model_from_config
from .dataset import EmbeddingSequenceDataset
from .utils import inference


def main():
    parser = argparse.ArgumentParser(description="Inference script for viscosity model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/viscosity_best_config.json",
        help="Path to JSON config file for the model hyperparameters",
    )
    parser.add_argument(
        "--properties_csv",
        type=str,
        required=True,
        help="Path to the property / embedding table CSV",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to the test dataset CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--mc_dropout",
        action="store_true",
        help="Use Monte Carlo Dropout for uncertainty estimation",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=20,
        help="Number of MC Dropout samples",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터 로딩
    properties = pd.read_csv(args.properties_csv)
    if "Unnamed: 0" in properties.columns:
        properties = properties.drop(columns=["Unnamed: 0"])

    test_data = pd.read_csv(args.test_csv)
    if "Unnamed: 0" in test_data.columns:
        test_data = test_data.drop(columns=["Unnamed: 0"])

    # 2) Dataset & DataLoader
    max_len = 15
    test_dataset = EmbeddingSequenceDataset(
        dat=test_data,
        embedding_table=properties,
        max_len=max_len,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 3) 모델 생성 + weight 로드
    input_dim = len(properties.iloc[:, 0]) + 2

    with open(args.config, "r") as f:
        config = json.load(f)

    model = build_viscosity_model_from_config(
        config=config,
        input_dim=input_dim,
        device=device,
    )

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # 4) 예측
    preds = inference(
        model=model,
        dataloader=test_loader,
        device=device,
        disable_permutation=True,
        return_target=False,
        mc_dropout=args.mc_dropout,
        T=args.T,
    )

    # 5) 저장

    if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 2:
        out_df = pd.DataFrame(
            {
                "prediction_mean": preds[:, 0],
                "prediction_std": preds[:, 1],
            }
        )
    else:
        out_df = pd.DataFrame({"prediction": preds})

    out_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
