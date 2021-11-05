from collections import defaultdict, deque
from typing import List
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
        self.backward_stack = []
        self._finish = False
        self._kept = {"_finish": True}

    def set_default(self, key, value, keep=False):
        if key not in self:
            self[key] = value
        if keep:
            self._kept[key] = True


class Task:

    def __init__(self) -> None:
        self.middleware = []
        # For eager mode
        self.ctx = Context(total_step=0)

    def use(self, fn) -> None:
        self.middleware.append(fn)

    def run(self, max_step=1e10):
        for _ in range(max_step):
            self.ctx = self.renew_ctx(self.ctx)
            self.run_one_step(self.ctx, self.middleware)
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
        r = g = fn(self.ctx)
        if isinstance(g, GeneratorType):
            r = next(g)
            self.ctx.backward_stack.append(g)
        return r

    def backward(self):
        for g in reversed(self.ctx.backward_stack):
            for _ in g:
                pass
        self.ctx = self.renew_ctx(self.ctx)

    def renew_ctx(self, ctx):
        new_ctx = Context(total_step=ctx.total_step + 1)
        for k in ctx._kept:
            new_ctx[k] = ctx[k]
        return new_ctx

    @property
    def finish(self):
        return self.ctx._finish


class DequeBuffer:

    def __init__(self, maxlen=20000) -> None:
        self.memory = deque(maxlen=maxlen)

    def push(self, data):
        self.memory.append(data)

    def sample(self, size):
        if size > len(self.memory):
            print('[Warning] no enough data: {}/{}'.format(size, len(self.memory)))
            return None
        indices = list(np.random.choice(a=len(self.memory), size=size, replace=False))
        return [self.memory[i] for i in indices]
        # return random.sample(self.memory, size)

    def count(self):
        return len(self.memory)


class DQNPipeline:

    def __init__(self, cfg, model, seed=0):
        self.cfg = cfg
        self.policy = DQNPolicy(cfg.policy, model=model)
        eps_cfg = cfg.policy.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def act(self, env):

        def _act(ctx):
            ctx.set_default("collect_env_step", 0, keep=True)
            eps = self.epsilon_greedy(ctx.collect_env_step)
            ctx.obs = env.ready_obs
            policy_output = self.policy.collect_mode.forward(ctx.obs, eps=eps)
            ctx.action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})
            ctx.policy_output = policy_output
            yield

        return _act

    def collect(self, env, buffer_):

        def _collect(ctx):
            timesteps = env.step(ctx.action)
            ctx.collect_env_step += len(timesteps)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, timestep in timesteps.items():
                transition = self.policy.collect_mode.process_transition(
                    ctx.obs[env_id], ctx.policy_output[env_id], timestep
                )
                buffer_.push(transition)
            yield

        return _collect

    def learn(self, buffer_):

        def _learn(ctx):
            ctx.set_default("train_iter", 0, keep=True)
            for i in range(self.cfg.policy.learn.update_per_collect):
                data = buffer_.sample(self.policy.learn_mode.get_attribute('batch_size'))
                if data is None:
                    break
                learn_output = self.policy.learn_mode.forward(data)
                if ctx.train_iter % 20 == 0:
                    print(
                        'Current Training: Train Iter({})\tLoss({:.3f})'.format(
                            ctx.train_iter, learn_output['total_loss']
                        )
                    )
                ctx.train_iter += 1
            yield

        return _learn

    def evaluate(self, env):

        def _eval(ctx):
            ctx.set_default("train_iter", 0, keep=True)
            ctx.set_default("last_eval_iter", -1, keep=True)
            if ctx.train_iter == ctx.last_eval_iter or ((ctx.train_iter - ctx.last_eval_iter) < self.cfg.policy.eval.evaluator.eval_freq and ctx.train_iter != 0):
                yield
                return
            env.reset()
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
            stop_flag = eval_reward >= self.cfg.env.stop_value and ctx.train_iter > 0
            print('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(ctx.train_iter, eval_reward))
            ctx.last_eval_iter = ctx.train_iter
            if stop_flag:
                ctx._finish = True
            yield

        return _eval


def speed_profile():
    buffer = defaultdict(list)

    def _speed_profile(fn, name):

        def _executor(ctx):
            # Execute before step's yield
            total_time = 0
            start = time.time()
            r = g = fn(ctx)
            if isinstance(g, GeneratorType):
                r = next(g)
            total_time += time.time() - start

            yield r

            # Execute after step's yield
            start = time.time()
            if isinstance(g, GeneratorType):
                for _ in g:
                    pass
            total_time += time.time() - start

            nonlocal buffer
            buffer[name].append(total_time)
            if ctx.total_step % 100 == 99:
                print("Total execute time for {}: {:.2f} ms/step".format(name, np.array(buffer[name]).mean() * 1000))
                buffer[name] = []

        return _executor

    return _speed_profile


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

    task.use(dqn.evaluate(evaluator_env))
    # for _ in range(8):
    task.use(dqn.act(collector_env))
    task.use(dqn.collect(collector_env, replay_buffer))
    task.use(dqn.learn(replay_buffer))

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

    evaluate = dqn.evaluate(evaluator_env)
    act = dqn.act(collector_env)
    collect = dqn.collect(collector_env, replay_buffer)
    learn = dqn.learn(replay_buffer)

    sp = speed_profile()

    for _ in range(1000):
        task.forward(evaluate)
        if task.finish:
            break
        task.forward(act)
        task.forward(sp(collect, "collect"))
        task.forward(sp(learn, "learn"))
        # Run rest logic and re init context
        task.backward()


if __name__ == "__main__":
    # main(main_config, create_config)
    main_eager(main_config, create_config)
