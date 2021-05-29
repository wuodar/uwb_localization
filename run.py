import os
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch.optim import Adam

from models.model import LinearModel
from constants import (
    RAW_TRAIN_DATA_PATHS,
    RAW_TEST_DATA_PATHS,
    COLUMNS,
    OUT_PATH,
)
from datasets.uwb_dataset import PosDataset
from utils import plot_cdf, plot_trajectory


class PosModel(LightningModule):
    def __init__(self, lr, batch_size, num_workers, max_error):
        super(PosModel, self).__init__()

        self.net = LinearModel()
        self.lr = lr
        self.max_error = max_error
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss = L1Loss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.net(x)
        loss = self.loss(out, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        out = self.net(x)
        loss = self.loss(out, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self.net(x)
        loss = self.loss(out, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def setup(self, stage):
        data = PosDataset(
            files=RAW_TRAIN_DATA_PATHS, max_error=self.max_error
        )  # + RAW_VAL_DATA_PATHS)
        train_size = int((len(data)) * 0.8)
        val_size = int(((len(data)) - train_size) / 2)
        test_size = len(data) - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(
            data, (train_size, val_size, test_size)
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=5e-5)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=64)

        return parser


if __name__ == "__main__":
    seed_everything(3)

    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"experiment{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
    )
    parser.add_argument("--max_error", type=float, default=5000)

    parser = PosModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    os.makedirs(OUT_PATH, exist_ok=True)
    
    experiment_dir = os.path.join(OUT_PATH, args.out_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    hparams = {
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "num_workers": 4,
        "max_error": args.max_error,
    }

    # Setup model and trainer
    model = PosModel(**hparams)
    trainer = Trainer(max_epochs=args.max_epochs, gpus=-1)

    trainer.fit(model)

    # save model to checkpoint
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pt"))
    model.eval()

    for file in RAW_TEST_DATA_PATHS:
        name = Path(file).name[:-5]
        data = pd.read_excel(
            file, usecols=COLUMNS
        )  # .values.astype(np.float32)
        x = torch.tensor(
            data[
                data.columns.difference(["reference__x", "reference__y"])
            ].values.astype(np.float32)
        )
        target = data[["reference__x", "reference__y"]].values.astype(
            np.float32
        )

        prediction = model(x).cpu().detach().numpy()

        plot_trajectory(
            data[["reference__x", "reference__y"]].to_numpy(),
            data[["data__coordinates__x", "data__coordinates__y"]].to_numpy(),
            prediction,
            experiment_dir,
            name=name,
        )

        model_err = np.linalg.norm(prediction - target, axis=1)
        raw_err = np.linalg.norm(
            data[["data__coordinates__x", "data__coordinates__y"]] - target,
            axis=1,
        )

        plot_cdf(raw_err, model_err, experiment_dir, name)
