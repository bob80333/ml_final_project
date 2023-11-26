import torch
import pytorch_metric_learning.losses as losses
import model
import dataset

OPTIM = torch.optim.AdamW
LR = 3e-4
BATCH_SIZE = 64

LOSS = losses.SupConLoss

TRAIN_STEPS = 10_000
EVAL_EVERY = 200

MODEL = model.TransformerModel

DATSET = dataset.CVSS_T
DATA_PATH = "data/preprocessed"
N_WORKERS = 2

KS = [1, 2, 4]  # for recall@k, (k = 1 is accuracy)

CHECKPOINT_PATH = "checkpoints/"
