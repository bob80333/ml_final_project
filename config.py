import torch
import pytorch_metric_learning.losses as losses
import model
import dataset

OPTIM = torch.optim.AdamW
LR = 3e-4
BATCH_SIZE = 8

LOSS = losses.SupConLoss

TRAIN_STEPS = 10_000
EVAL_EVERY = 200

MODEL = model.TransformerModel

DATASET = dataset.CVSS_T
DATA_PATH = "data/preprocessed_10k"
N_WORKERS = 0

KS = [1, 2, 4]  # for recall@k, (k = 1 is accuracy)

CHECKPOINT_PATH = "checkpoints/"
