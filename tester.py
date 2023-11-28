import torch

import config
import evaluation


class Tester:
    def __init__(self, device, checkpoint_path):
        self.device = device
        self.model = torch.compile(config.MODEL().to(device))
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        self.test_dataset = config.DATA_PATH + "/test"

    def test(self):
        test_loader = torch.utils.data.DataLoader(
            config.DATASET(self.test_dataset, random_segment=False),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.N_WORKERS,
        )

        accuracy = evaluation.evaluate_r_at_k(test_loader, self.model, self.device, config.KS)
        
        print(f"Accuracy (R@1): {accuracy}")