from collections import deque
import gym
import random
import pickle
import torch
import time
import numpy as np
from types import GeneratorType
from rich import print
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.config import compile_config
from ding.policy import DQNPolicy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.torch_utils import to_ndarray, to_tensor
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


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
            policy_output=None,
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
        self.policy_output = policy_output


class Task:

    def __init__(self) -> None:
        self.middleware = []
        self.finish = False
        self.collect_env_step = 0
        self.train_iter = 0
        self.last_eval_iter = -1
        # For eager mode
        self.stack = []
        self.total_step = 0
        self.ctx = Context(total_step=0)

    def use(self, fn) -> None:
        self.middleware.append(fn)

    def run(self, max_step=1e10):
        for i in range(max_step):
            ctx = Context(total_step=i)  # Initial context
            self.run_one_step(ctx, self.middleware)
            if self.finish:
                break

    def run_one_step(self, ctx, middleware):
        if len(middleware) == 0:
            return

        stack = []
        for fn in middleware:
            g = fn(ctx)
            if isinstance(g, GeneratorType):
                next(g)
                stack.append(g)
        # TODO how to treat multiple yield
        # TODO how to use return value or send value to generator
        # Loop over logic after yield
        for g in reversed(stack):
            for _ in g:
                pass

    def forward(self, fn):
        g = fn(self.ctx)
        if isinstance(g, GeneratorType):
            next(g)
            self.stack.append(g)
        return self.ctx

    def backward(self):
        for g in reversed(self.stack):
            for _ in g:
                pass
        self.stack = []
        self.total_step += 1
        self.ctx = Context(total_step=self.total_step)


class DequeBuffer:

    def __init__(self, maxlen=10000) -> None:
        self.memory = deque(maxlen=maxlen)

    def push(self, data):
        self.memory.append(data)

    def sample(self, size):
        if size > len(self.memory):
            print('[Warning] no enough data: {}/{}'.format(size, len(self.memory)))
            return None
        return random.sample(self.memory, size)

    def count(self):
        return len(self.memory)


class DQNPipeline:

    def __init__(self, cfg, model, seed=0):
        self.cfg = cfg
        self.policy = DQNPolicy(cfg.policy, model=model)
        eps_cfg = cfg.policy.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def act(self, env, task):

        def _act(ctx):
            eps = self.epsilon_greedy(ctx.step)
            ctx.obs = env.ready_obs
            policy_output = self.policy.collect_mode.forward(ctx.obs, eps=eps)
            ctx.action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})
            ctx.policy_output = policy_output
            yield

        return _act

    def collect(self, env, buffer_, task):

        def _collect(ctx):
            timesteps = env.step(ctx.action)
            task.collect_env_step += len(timesteps)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, timestep in timesteps.items():
                transition = self.policy.collect_mode.process_transition(
                    ctx.obs[env_id], ctx.policy_output[env_id], timestep
                )
                buffer_.push(transition)
            yield

        return _collect

    def learn(self, buffer_, task):

        def _learn(ctx):
            for i in range(self.cfg.policy.learn.update_per_collect):
                data = buffer_.sample(self.policy.learn_mode.get_attribute('batch_size'))
                if data is None:
                    break
                learn_output = self.policy.learn_mode.forward(data)
                if task.train_iter % 20 == 0:
                    print(
                        'Current Training: Train Iter({})\tLoss({:.3f})'.format(
                            task.train_iter, learn_output['total_loss']
                        )
                    )
                task.train_iter += 1
            yield

        return _learn

    def evaluate(self, env, task):

        def _eval(ctx):
            eval_interval = task.train_iter - task.last_eval_iter
            if eval_interval < self.cfg.policy.eval.evaluator.eval_freq:
                yield
                return
            eval_monitor = VectorEvalMonitor(env.env_num, self.cfg.env.n_evaluator_episode)
            while not eval_monitor.is_finished():
                obs = env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self.policy.eval_mode.forward(obs)
                action = to_ndarray({i: a['action'] for i, a in policy_output.items()})
                timesteps = env.step(action)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, timestep in timesteps.items():
                    if timestep.done:
                        self.policy.eval_mode.reset([env_id])
                        reward = timestep.info['final_eval_reward']
                        eval_monitor.update_reward(env_id, reward)
            episode_reward = eval_monitor.get_episode_reward()
            eval_reward = np.mean(episode_reward)
            stop_flag = eval_reward >= self.cfg.env.stop_value and task.train_iter > 0
            print('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(task.train_iter, eval_reward))
            task.last_eval_iter = task.train_iter
            if stop_flag:
                task.finish = True
            yield

        return _eval


def main(cfg, create_cfg, seed=0):

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    cfg = compile_config(cfg, create_cfg=create_cfg, auto=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    collector_env.launch()
    evaluator_env.launch()

    model = DQN(**cfg.policy.model)
    replay_buffer = DequeBuffer()

    task = Task()
    dqn = DQNPipeline(cfg, model)

    task.use(dqn.evaluate(evaluator_env, task))
    # for _ in range(8):
    task.use(dqn.act(collector_env, task))
    task.use(dqn.collect(collector_env, replay_buffer, task))
    task.use(dqn.learn(replay_buffer, task))

    task.run(max_step=1000)


def main_eager(cfg, create_cfg, seed=0):

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    cfg = compile_config(cfg, create_cfg=create_cfg, auto=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    collector_env.launch()
    evaluator_env.launch()

    model = DQN(**cfg.policy.model)
    replay_buffer = DequeBuffer()

    task = Task()
    dqn = DQNPipeline(cfg, model)

    evaluate = dqn.evaluate(evaluator_env, task)
    act = dqn.act(collector_env, task)
    collect = dqn.collect(collector_env, replay_buffer, task)
    learn = dqn.learn(replay_buffer, task)
    for _ in range(1000):
        task.forward(evaluate)
        task.forward(act)
        task.forward(collect)
        task.forward(learn)
        # Run rest logic and re init context
        task.backward()
        if task.finish:
            break


if __name__ == "__main__":
    # main(main_config, create_config)
    main_eager(main_config, create_config)
