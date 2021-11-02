from collections import deque
import gym
import random
import torch
import time
import numpy as np
from rich import print
import os
import gym
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config


class Context(dict):

    def __init__(
            self,
            total_step=0,  # Total steps
            step=0,  # Step in current episode
            episode=0,
            state=None,
            next_state=None,
            action=None,
            reward=None,
            done=False,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.total_step = total_step
        self.step = step
        self.episode = episode
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done


class Task:

    def __init__(self, env) -> None:
        self.middleware = []
        self.env = env
        self.finish = False

    def use(self, fn) -> None:
        self.middleware.append(fn)

    def run(self, max_step):
        state = self.env.reset()
        ctx = Context(state=state)  # Initial context
        while not self.finish:
            self.run_one_step(ctx, self.middleware)
            if ctx.done:
                ctx = Context(total_step=ctx.total_step + 1, step=0, episode=ctx.episode + 1, state=self.env.reset())
            else:
                ctx = Context(
                    total_step=ctx.total_step + 1, step=ctx.step + 1, episode=ctx.episode, state=ctx.next_state
                )
            if ctx.total_step >= max_step:
                self.finish = True

    def run_one_step(self, ctx, middleware):
        if len(middleware) == 0:
            return

        def chain():
            self.run_one_step(ctx, middleware[1:])

        middleware[0](ctx, chain)


class DequeBuffer:

    def __init__(self, maxlen=2000) -> None:
        self.memory = deque(maxlen=maxlen)

    def push(self, data):
        self.memory.append(data)

    def sample(self, size):
        return random.sample(self.memory, size)


# Get DI-engine form env class
def wrapped_cartpole_env():
    return DingEnvWrapper(gym.make('CartPole-v0'))


def main(cfg, seed=0):
    # Initial
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    # Set random seed for all package and instance
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    print(replay_buffer)
    exit()

    task = Task(env)
    dqn = DQN()

    task.use(dqn.main())

    task.run(max_step=100000)


if __name__ == "__main__":
    main(cartpole_dqn_config)
