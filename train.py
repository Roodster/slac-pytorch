import argparse
import os
from datetime import datetime

import torch

from slac_pytorch.algo import SlacAlgorithm
from slac_pytorch.env import make_dmc
from slac_pytorch.trainer import Trainer
from slac_pytorch.common.utils import parse_args, save_config

def main(args):
    env = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )
    env_test = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )

    parameters_dir = os.path.join(
        f"{args.working_dir}logs/parameters/",
        f"{args.domain_name}-{args.task_name}",
        f'slac-{args.domain_name}-{args.task_name}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    )
    
    save_config(args, parameters_dir)

    log_dir = os.path.join(
        f"{args.working_dir}logs/runs/",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        args=args
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args(args_file="./data/configs/default.json")
    main(args)