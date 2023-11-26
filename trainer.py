import config
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
            shuffle=True,
            num_workers=config.N_WORKERS,
            persistent_workers=True,
        )

        train_loader = infinite_data_generator(train_loader)

        pbar = trange(config.TRAIN_STEPS)
        for step in pbar:
            batch = next(train_loader)
            loss = self.train_step(batch)

            pbar.set_description(f"Step {step}, loss: {loss}")

            if step % config.EVAL_EVERY == 0:
                print(f"Step {step}, loss: {loss}")
                self.eval(val_loader)
                torch.save(self.model.state_dict(), f"model_{step}.pt")
                self.model.train()

    def eval(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            all_german_embeds = []
            all_english_embeds = []
            for batch in dataloader:
                english_audio, german_audio, idx = batch
                english_audio = english_audio.to(self.device)
                german_audio = german_audio.to(self.device)
                ids = idx.to(self.device)

                german_embeds = self.model(german_audio)
                english_embeds = self.model(english_audio)

                all_german_embeds.append(german_embeds)
                all_english_embeds.append(english_embeds)

            all_german_embeds = torch.cat(all_german_embeds, dim=0)
            all_english_embeds = torch.cat(all_english_embeds, dim=0)

            # calculate r@k
            r_values = []
            for k in config.KS:
                r = 0
                for i in range(
                    0,
                    len(all_german_embeds),
                ):
                    # get the embeddings of the german audio
                    german_embed = all_german_embeds[i]
                    # calculate the cosine similarity between the english audio and all the german audio
                    cosine_similarities = torch.nn.functional.cosine_similarity(
                        all_english_embeds, german_embed
                    )
                    # get the indices of the top k most similar german audio
                    top_k = cosine_similarities.topk(
                        k=k, largest=True, sorted=True
                    ).indices
                    # check if the index of the first english audio is in the top k most similar german audio
                    if i in top_k:
                        r += 1
                r /= len(all_german_embeds) / 2
                r_values.append(r)

            print(f"r@{config.KS}: {r_values}")
            return r_values
