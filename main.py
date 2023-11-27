import trainer
import tester
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", help="Training device", choices=["cpu", "cuda"])
    parser.add_argument("--train", help="train the model", action="store_true")
    parser.add_argument("--test", help="test the model", action="store_true")
    args = parser.parse_args()
    
    if args.train or args.test:
        # make sure checkpoint dir exists
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

    if args.train:
        print("Training the model")
        trainer = trainer.Trainer(args.device)
        trainer.train()
    elif args.test:
        print("Testing the model")
        tester = tester.Tester(args.device, "checkpoints/best_model.pt")
        tester.test()
    else:
        print("Please specify either --train or --test, neither specified so nothing to do.")