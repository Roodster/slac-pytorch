from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import gym.spaces
import numpy as np
from torchrl.envs import GymWrapper


class RenderGymWrapper(GymWrapper):

  def __init__(self, gym_env, render_kwargs=None):
    super(RenderGymWrapper, self).__init__(gym_env)
    self._render_kwargs = dict(
        width=64,
        height=64,
        depth=False,
        camera_name='track',
    )
    if render_kwargs is not None:
      self._render_kwargs.update(render_kwargs)

  @property
  def sim(self):
    return self._env.sim

  def render(self, mode='rgb_array'):
    if mode == 'rgb_array':
      return self._env.sim.render(**self._render_kwargs)[::-1, :, :]
    else:
      return self._env.render(mode=mode)


class PixelObservationsGymWrapper(GymWrapper):

  def __init__(self, gym_env, observations_whitelist=None, render_kwargs=None):
    super(PixelObservationsGymWrapper, self).__init__(gym_env)
    if observations_whitelist is None:
      self._observations_whitelist = ['state', 'pixels']
    else:
      self._observations_whitelist = observations_whitelist
    self._render_kwargs = dict(
        width=64,
        height=64,
        depth=False,
        camera_name='track',
    )
    if render_kwargs is not None:
      self._render_kwargs.update(render_kwargs)

    observation_spaces = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observation_spaces['state'] = self._env.observation_space
      elif observation_name == 'pixels':
        image_shape = (
          self._render_kwargs['height'], self._render_kwargs['width'], 3)
        image_space = gym.spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        observation_spaces['pixels'] = image_space
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    self.observation_space = gym.spaces.Dict(observation_spaces)

  def _modify_observation(self, observation):
    observations = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observations['state'] = observation
      elif observation_name == 'pixels':
        image = self._env.sim.render(**self._render_kwargs)[::-1, :, :]
        observations['pixels'] = image
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    observation = self._modify_observation(observation)
    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)

  def render(self, mode='rgb_array'):
    return self._env.render(mode=mode)




class VideoWrapper(GymWrapper):
  def __init__(self, env):
    """Wrapper that keeps track of frames that are rendered after every step.

    It is useful when this environment is wrapped by an ActionRepeat wrapper,
    in which the latter would not be able to render the frames in between steps.
    The `frames` buffer is unbounded and it should be periodically emptied
    elsewhere outside of this class.
    """
    super(VideoWrapper, self).__init__(env)
    self._frames = []
    self._rendering = False

  def _reset(self):
    time_step = self._env.reset()
    if self._rendering:
      self._frames.append(self._env.render())
    return time_step

  def _step(self, action):
    time_step = self._env.step(action)
    if self._rendering:
      self._frames.append(self._env.render())
    return time_step

  @property
  def frames(self):
    return self._frames

  @property
  def rendering(self):
    return self._rendering

  def start_rendering(self):
    self._rendering = True

  def stop_rendering(self):
    self._rendering = False