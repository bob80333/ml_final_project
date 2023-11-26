from tabnanny import check
import config
import evaluation
import torch

class Tester:
    def __init__(self, device, checkpoint_path):
        self.device = device
        self.model = config.MODEL().to(device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        self.test_dataset = config.DATA_PATH + "/test"

    def test(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.N_WORKERS,
        )

        accuracy = evaluation.evaluate_r_at_k(test_loader, self.model, self.device, config.KS)
        
        print(f"Accuracy (R@1): {accuracy}")