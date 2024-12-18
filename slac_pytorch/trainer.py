import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        # print('appending state to obs: ', state.shape)

        self._state.append(state)
        self._action.append(action)
        # print("Current state: \n ", self._state)

    @property
    def state(self):
        # print('CALLED STATE: \n', self._state)
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        envs,
        env,
        env_test,
        algo,
        log_dir,
        current_steps=1,
        args=None,
    ):
        assert args is not None
        # Env to collect samples.
        self.envs = envs
        
        for env in self.envs:
            _ = env.reset(seed=args.seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.reset(seed=2 ** 31 - args.seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, args.num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, args.num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = args.action_repeat
        self.num_steps = int(args.num_steps)
        self.initial_collection_steps = int(args.initial_collection_steps)
        self.initial_learning_steps = int(args.initial_learning_steps)
        self.eval_interval = int(args.eval_interval)
        self.num_eval_episodes = int(args.eval_num_episodes)
        self.current_step = current_steps

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        env_id = 0
        # Initialize the environment.
        state, _ = self.envs[env_id].reset()
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        bar = tqdm(range(1, self.initial_collection_steps + 1))
        for step in bar:
            bar.set_description("Collecting trajectories using random policy.")
            t = self.algo.step(self.envs[env_id], self.ob, t, step <= self.initial_collection_steps)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        
        bar = tqdm(range(self.current_step, self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        start_env_steps = self.initial_collection_steps + 1 if self.current_step == 1 else self.current_step
        bar = tqdm(range(start_env_steps, start_env_steps + self.num_steps // self.action_repeat + 1))
        for step in bar:

            
            if t == 0:
                env_id = np.random.choice(list(range(len(self.envs)))) - 1
                
            t = self.algo.step(self.envs[env_id], self.ob, t, False)
            
            # if t is 0 the episode is over and we sample a next environment to simulate in.

            # Update the algorithm.
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                mean_return = self.evaluate(step_env)
                bar.set_description(f"iter={step} mean_return={mean_return}")
                self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))
                self.current_step = step

    def evaluate(self, step_env):
        mean_return = 0.0

        for i in range(self.num_eval_episodes):
            state, _ = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                state, reward, terminated, truncated, infos = self.env_test.step(action)
                done = terminated or truncated
                self.ob_test.append(state, action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, mode='w', index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        return mean_return

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
