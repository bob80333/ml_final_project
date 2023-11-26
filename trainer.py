from sklearn.base import accuracy_score
import config
import evaluation
import torch
from tqdm import trange


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

        self.train_dataset = config.DATA_PATH + "/train"
        self.val_dataset = config.DATA_PATH + "/val"

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        english_audio, german_audio, idx = batch
        english_audio = english_audio.to(self.device)
        german_audio = german_audio.to(self.device)
        ids = idx.to(self.device)

        all_audio = torch.cat((english_audio, german_audio), dim=0)
        all_ids = ids.repeat(2)

        all_embeds = self.model(all_audio)
        loss = self.criterion(all_embeds, all_ids)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.N_WORKERS,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.N_WORKERS,
            persistent_workers=True,
        )

        train_loader = infinite_data_generator(train_loader)

        pbar = trange(config.TRAIN_STEPS)
        
        # save values for plotting
        loss_values = torch.empty(config.TRAIN_STEPS)
        accuracy_values = torch.ones(config.TRAIN_STEPS) * -1 # -1 for non evaluated steps, so we can remove them later when plotting
        
        # save best model
        best_accuracy = -1
        for step in pbar:
            batch = next(train_loader)
            loss = self.train_step(batch)

            pbar.set_description(f"Step {step}, loss: {loss}")
            loss_values[step] = loss
            if step % config.EVAL_EVERY == 0:
                print(f"Step {step}, loss: {loss}")
                accuracy = self.eval(val_loader)
                accuracy_values[step] = accuracy
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    torch.save(self.model.state_dict(), f"{config.CHECKPOINT_PATH}/best_model.pt")
                torch.save(self.model.state_dict(), f"{config.CHECKPOINT_PATH}/model_{step}.pt")
                self.model.train()
                
                
        print("Training complete!")
        
        # save loss and accuracy values for plotting later
        torch.save(loss_values, f"{config.CHECKPOINT_PATH}/loss_values.pt")
        torch.save(accuracy_values, f"{config.CHECKPOINT_PATH}/accuracy_values.pt")

    def eval(self, dataloader):
        # return accuracy (k=1)
        return evaluation.evaluate_r_at_k(dataloader, self.model, self.device, config.KS)[0]

    