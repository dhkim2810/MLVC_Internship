import os
import torch
from tqdm import trange
from argparse import ArgumentParser

import ray
from ray.util.sgd import TorchTrainer
from module import CIFAR10Module


def main(args):
    if args.smoke_test:
        ray.init(num_cpus=4)
    else:
        ray.init(address=args.address, num_cpus=args.num_workers, log_to_driver=True)

    # Trainer Initialization
    trainer = TorchTrainer(
        training_operator_cls=CIFAR10Module,
        num_workers=args.num_workers,
        config={
            "lr" : args.learning_rate,
            "momentum" : args.momentum,
            "data_dir" : args.data_dir,
            "batch_size" : args.batch_size,
            "num_workers" : args.num_workers,
            "smoke_test" : args.smoke_test
        },
        use_gpu=args.use_gpu,
        scheduler_step_freq="epoch",
        use_fp16=args.fp16,
        use_tqdm=False)

    train_loss = []
    val_loss = []
    val_acc = []

    path = os.path.join("/root/volume/Paper/MLVC_Internship", args.checkpoint_dir, args.model_name+"_"+str(args.trial))
    if not os.path.exists(path):
        os.mkdir(path)

    from tabulate import tabulate
    pbar = trange(args.max_epochs, unit="epoch")
    for it in pbar:
        stats = trainer.train(max_retries=1, info=dict(epoch_idx=it, num_epochs=args.max_epochs))
        train_loss.append(stats["train_loss"])
        val_stats = trainer.validate()
        val_loss.append(val_stats["val_loss"])
        pbar.set_postfix(dict(acc=val_stats["val_accuracy"]))

        trainer.save("/root/volume/Paper/MLVC_Internship/checkpoint/{}_{}/epoch_{}.ray".format(args.model_name, args.trial, it))
        torch.save([train_loss, val_loss],"/root/volume/Paper/MLVC_Internship/checkpoint/{}_{}/epoch_{}.loss".format(args.model_name, args.trial, it))
        torch.save([val_acc],"/root/volume/Paper/MLVC_Internship/checkpoint/{}_{}/epoch_{}.acc".format(args.model_name, args.trial, it))

    print(val_stats)
    trainer.shutdown()
    print("success!")


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument(
        "--data_dir",
        type=str, 
        default="/dataset/CIFAR")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables training with GPU"
    )
    parser.add_argument(
        "--smoke-test", 
        action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address", 
        required=False, 
        type=str, help="the address to use to connect to a cluster.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enables FP16 training with apex. Requires `use-gpu`.")

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=False)

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--trial", type=int, default=1)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)