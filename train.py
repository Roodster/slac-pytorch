import argparse
import os
from datetime import datetime

import torch

from slac_pytorch.common.xml_manager import XML
from slac_pytorch.algo import SlacAlgorithm, ObsSlacAlgorithm
from slac_pytorch.env import make_dmc, make_gym
from slac_pytorch.trainer import Trainer
from slac_pytorch.common.utils import parse_args, save_config
from slac_pytorch.environments.wrappers import AntImageWrapper


def main(args, universe='gym'):
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

        if args.universe == 'gym':
            
            env = make_gym(
                env=args.domain_name,
                action_repeat=args.action_repeat,
                render_mode=args.render_mode,
                environment_kwargs=dict(
                    xml_file=args.agent_path
                )
            )
            
            # Wrap the environment with our custom wrapper
            env = AntImageWrapper(env, image_size=(64, 64))
                        
        else:

            env = make_dmc(
                domain_name=args.domain_name,
                task_name=args.task_name,
                action_repeat=args.action_repeat,
                from_pixels=True,
                image_size=64,
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
        
        
        if args.universe == 'gym':
            env_test = make_gym(
                env=args.domain_name,
                action_repeat=args.action_repeat,
                render_mode=args.render_mode,
                environment_kwargs=dict(
                    xml_file=args.agent_path
                )
            )
            # Wrap the environment with our custom wrapper
            env_test = AntImageWrapper(env_test, image_size=(64, 64))
            
        else:
        
            env_test = make_dmc(
                domain_name=args.domain_name,
                task_name=args.task_name,
                action_repeat=args.action_repeat,
                from_pixels=True,
                image_size=64,
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
        f'slac-beta{args.beta}-seed{args.seed}',
    )


    algo = SlacAlgorithm(
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
