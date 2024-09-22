import argparse
import os
from datetime import datetime

import torch

from slac_pytorch.common.xml_manager import XML
from slac_pytorch.algo import SlacAlgorithm, ObsSlacAlgorithm
from slac_pytorch.env import make_dmc
from slac_pytorch.trainer import Trainer
from slac_pytorch.common.utils import parse_args, save_config

def main(args):
    masses = [750, 750, 1250, 1250]
    frictions = [0.5, 1.1, 0.5, 1.1]
    
    pairs = list(zip(masses, frictions))
    envs = []
    xml = XML()
    
    for pair in pairs[:-1]:
        mass, friction = pair
        
        values = dict(mass=mass, 
                      friction=friction)
        
        xml.modify(input_file=args.agent_path, output_file=args.agent_path ,values=values)

        env = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            from_pixels=False,
            # from_pixels=True,
            # image_size=64,
            environment_kwargs=dict(
                agent_path=args.agent_path
            )
        )
        envs.append(env)
        
    for pair in pairs[-1:]:

        mass, friction = pair
                
        values = dict(mass=mass, 
                    friction=friction)
        
        xml.modify(input_file=args.agent_path, output_file=args.agent_path ,values=values)
        
        env_test = make_dmc(
            domain_name=args.domain_name,
            task_name=args.task_name,
            action_repeat=args.action_repeat,
            from_pixels=False,
            # image_size=64,
            environment_kwargs=dict(
                agent_path=args.agent_path
            )
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


    algo = ObsSlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        args=args
    )
    trainer = Trainer(
        envs=envs,
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
