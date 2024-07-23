import dmc2gym
import gymnasium as gym

gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, environment_kwargs=None, image_size=64):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
        environment_kwargs=environment_kwargs
    )
    setattr(env, 'action_repeat', action_repeat)
    return env
