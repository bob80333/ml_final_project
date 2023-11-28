import pytorch_metric_learning.losses as losses
import torch

import dataset
import model

OPTIM = torch.optim.AdamW
LR = 1e-4
BATCH_SIZE = 512

LOSS = losses.SupConLoss

TRAIN_STEPS = 50_001
EVAL_EVERY = 2_000

MODEL = model.TransformerModel

DATASET = dataset.CVSS_T
DATA_PATH = "data/preprocessed"
N_WORKERS = 8

KS = [1, 2, 4, 8]  # for recall@k, (k = 1 is accuracy)

CHECKPOINT_PATH = "checkpoints/"
