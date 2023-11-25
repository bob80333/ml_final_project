import config
import torch


def infinite_data_generator(loader):
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, device):
        self.device = device
        self.model = config.MODEL().to(device)
        self.optimizer = config.OPTIM(self.model.parameters(), lr=config.LR)
        self.criterion = config.LOSS()
        
        self.train_dataset = config.TRAIN_DATASET
        self.val_dataset = config.VAL_DATASET
        
        
        
    def train_step(self, batch):
        
        self.model.train()
        self.optimizer.zero_grad()
        
        english_audio, german_audio, idx = batch
        english_audio = english_audio.to(self.device)
        german_audio = german_audio.to(self.device)
        ids = idx.to(self.device)
        
        all_audio = torch.cat((english_audio, german_audio), dim=0)
        all_ids = torch.repeat(ids, 2)
        
        all_embeds = self.model(all_audio)
        loss = self.criterion(all_embeds, all_ids)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
        
        
    def train(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.N_WORKERS)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.N_WORKERS)

        train_loader = infinite_data_generator(train_loader)
        
        for step in range(config.TRAIN_STEPS):
            batch = next(train_loader)
            loss = self.train_step(batch)
            
            if step % config.EVAL_EVERY == 0:
                print(f"Step {step}, loss: {loss}")
                self.eval()
                self.model.train()