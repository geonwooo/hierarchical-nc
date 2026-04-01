import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import argparse
import numpy as np

import _init_paths
import dataset as custom_dataset
from builder import Network
from config import cfg, update_config
from core.function import test_model
from core.evaluate import accuracy, AverageMeter
from data_transform.transform_wrapper import get_transform
from utils.reprod import fix_seed
from utils.dist import setup, cleanup


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main_worker(rank, world_size, args):
    # ----- BEGIN basic setting -----
    update_config(cfg, args)

    verbose = True if not cfg.ddp else True if rank == 0 else False

    fix_seed(cfg.seed_num)

    if cfg.ddp:
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size, port=cfg.port)

    torch.cuda.set_device(rank)
    # ----- END basic setting -----

    # ----- BEGIN dataset setting -----
    transform_ts = get_transform(cfg, mode='test')

    test_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=False, download=True, transform=transform_ts)
    if not isinstance(test_set.targets, torch.Tensor):
        test_set.targets = torch.tensor(test_set.targets, dtype=torch.long)
    num_classes = len(torch.unique(test_set.targets))
    
    testsampler = DistributedSampler(test_set, shuffle=False) if cfg.ddp else None
    
    if cfg.ddp:
        batch_size = int(cfg.train.batch_size / world_size)
        num_workers = int((cfg.train.num_workers+world_size-1)/world_size)
    else:
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers

    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        sampler=testsampler,
    )
    # ----- END dataset setting -----

    test_model(
        testloader, cfg, rank, verbose,
        num_classes=num_classes, pretrained=cfg.pretrained)

    if cfg.ddp:
        cleanup()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    if cfg.ddp:
        ngpus_per_node = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        rank = cfg.rank if cfg.rank != -1 else 0
        main_worker(rank, cfg.world_size, args)

