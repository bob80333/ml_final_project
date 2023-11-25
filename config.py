import torch
import pytorch_metric_learning
import model

OPTIM = torch.optim.AdamW
LR = 3e-4
LOSS = pytorch_metric_learning.losses.SupConLoss
BATCH_SIZE = 1024
TRAIN_STEPS = 100_000
EVAL_EVERY = 2_000
MODEL = model.TransformerModel
DATA_PATH = "data/preprocessed"
N_WORKERS = 4