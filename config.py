import torch
import pytorch_metric_learning.losses as losses
import model
import dataset

OPTIM = torch.optim.AdamW
LR = 1e-4
BATCH_SIZE = 256

LOSS = losses.SupConLoss

TRAIN_STEPS = 50_000
EVAL_EVERY = 2_000

MODEL = model.TransformerModel

DATASET = dataset.CVSS_T
DATA_PATH = "data/preprocessed"
N_WORKERS = 8

KS = [1, 2, 4]  # for recall@k, (k = 1 is accuracy)

CHECKPOINT_PATH = "checkpoints/"
