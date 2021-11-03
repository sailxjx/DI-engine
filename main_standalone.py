from collections import deque
from typing import List
import gym
import random
import pickle
import torch
import time
import numpy as np
from rich import print


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

    def __init__(self) -> None:
        self.middleware = []
        self.finish = False

    def use(self, fn) -> None:
        if isinstance(fn, List):
            self.middleware += fn
        else:
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

    def count(self):
        return len(self.memory)


def get_model(state_size, action_size):
    return torch.nn.Sequential(
        torch.nn.Linear(state_size, 24), torch.nn.ReLU(), torch.nn.Linear(24, 24), torch.nn.ReLU(),
        torch.nn.Linear(24, action_size)
    )


class DQN:

    def __init__(self) -> None:
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 64

    def middleware(self, model, action_size, env, max_step, buffer):
        return [
            self.act(model, action_size),
            self.collect(env, max_step=max_step),
            self.memorize(buffer),
            self.replay(model, buffer)
        ]

    def act(self, model, action_size):

        def _act(ctx, chain):
            if np.random.rand() < self.epsilon:
                ctx.action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    act_values = model(torch.tensor(ctx.state).float().reshape(1, -1))[0]
                    ctx.action = act_values.argmax().item()
            chain()

        return _act

    def collect(self, env, max_step):

        def _collect(ctx, chain):
            ctx.next_state, ctx.reward, ctx.done, _ = env.step(ctx.action)
            chain()
            if ctx.step >= max_step:
                ctx.done = True

        return _collect

    def memorize(self, buffer):

        def _memorize(ctx, chain):
            buffer.push((ctx.state, ctx.action, ctx.reward, ctx.next_state, ctx.done))
            chain()

        return _memorize

    def replay(self, model, buffer):
        loss_fn = torch.nn.MSELoss(reduction="mean")
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def _replay(ctx, chain):
            if ctx.total_step < self.batch_size:
                return chain()

            minibatch = buffer.sample(self.batch_size)
            states, targets_f = [], []
            # TODO Use tensor based calculation
            with torch.no_grad():
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        next_state = torch.tensor(next_state).float().reshape(1, -1)
                        target = reward + self.gamma * model(next_state)[0].max().item()
                    target_f = model(torch.tensor(state).float().reshape(1, -1)).numpy()
                    target_f[0][action] = target
                    targets_f.append(target_f)
                    states.append(state)

            targets_f = torch.tensor(np.array(targets_f)).reshape(len(minibatch), -1)
            states = torch.tensor(states).float()
            targets_hat = model(states)

            loss = loss_fn(targets_hat, targets_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if ctx.done:
                template = '[bright_blue]'
                template += 'Episode: {:>3d} '
                template += '| Reward: {:>3d} '
                template += '| Total: {:<4d} '
                template += '| Loss: {:>6.5f} '
                template += '| Epsilon: {:>3.2f}'
                template += '[/bright_blue]'
                print(template.format(ctx.episode, ctx.step, ctx.total_step, loss.item(), self.epsilon))

            chain()

        return _replay


class SpeedProfile():

    def __init__(self, name="", eval_step=500) -> None:
        self.name = name
        self.eval_step = eval_step
        self.forward = deque(maxlen=self.eval_step)
        self.backward = deque(maxlen=self.eval_step)
        self.forward_start = 0
        self.backward_start = 0
        self.last_start = time.time()

    def before(self, ctx, chain):
        self.forward_start = time.time()
        chain()
        self.backward.append(time.time() - self.backward_start)

    def after(self, ctx, chain):
        self.forward.append(time.time() - self.forward_start)
        chain()
        self.backward_start = time.time()
        if ctx.total_step and ctx.total_step % self.eval_step == 0:
            duration = time.time() - self.last_start
            self.last_start = time.time()
            forward = np.array(self.forward) * 1000
            backward = np.array(self.backward) * 1000
            total = forward + backward
            template = '[grey53]'
            template += 'SpeedProfile {} (mean+std) ms/step: '
            template += 'forward: {:>.2f}+{:<.2f}, '
            template += 'backward: {:>.2f}+{:<.2f}, '
            template += 'total: {:>.2f}+{:<.2f}, '
            template += 'step/s: {:>.0f}'
            template += '[/grey53]'
            print(
                template.format(
                    self.name,\
                    np.mean(forward), np.std(forward),\
                    np.mean(backward), np.std(backward),\
                    np.mean(total), np.std(total),\
                    self.eval_step / duration
                ))


def reset_env(env):
    state, step, episode = None, 0, 0

    def _reset_env(ctx, chain):
        nonlocal state, step, episode
        ctx.state = env.reset() if state is None else state
        ctx.step = step
        ctx.episode = episode

        chain()

        if ctx.done:
            state = None
            episode += 1
            step = 0
        else:
            state = ctx.next_state
            step += 1

    return _reset_env


def save_and_quit(task, buffer, model, max_reward, n_episode):
    '''
    Save checkpoint and data if n_episode's reward is greater than max_reward
    '''
    n_win = 0

    def save_data():
        states = [d[0] for d in buffer.sample(buffer.count())]
        data = {"states": torch.tensor(states).float()}
        with torch.no_grad():
            data["q"] = model(data["states"])
        with open("tmp/data.pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _save_and_quit(ctx, chain):
        chain()
        nonlocal n_win
        if ctx.done:
            if max_reward <= ctx.step:
                n_win += 1
            else:
                n_win = 0
            if n_win >= n_episode:
                save_data()
                template = "[green]Total reward is greater than {} in the lastest {} episodes, final episode {}, bye.[/green]"
                print(template.format(max_reward, n_episode, ctx.episode))
                task.finish = True

    return _save_and_quit


def train_with_expert(model, size=64):
    with open("tmp/data.pkl", "rb") as f:
        data = pickle.load(f)
    states = data["states"]
    q = data["q"]

    loss_fn = torch.nn.MSELoss(reduction="mean")
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _train_with_expert(ctx, chain):
        indice = random.sample(range(len(q)), k=size)
        _states = states[indice]
        _q = q[indice]
        _q_hat = model(_states)
        loss = loss_fn(_q_hat, _q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        chain()

    return _train_with_expert


def main():
    # Initial
    env = gym.make("CartPole-v0")
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    model = get_model(state_size, action_size)
    buffer = DequeBuffer()
    speed_profile = SpeedProfile("replay")

    task = Task()
    dqn = DQN()

    task.use(reset_env(env))
    # task.use(dqn.middleware(model, action_size, env, 500, buffer))
    task.use(dqn.act(model, action_size))
    task.use(dqn.collect(env, max_step=500))
    task.use(dqn.memorize(buffer))

    task.use(train_with_expert(model))

    task.use(speed_profile.before)
    task.use(dqn.replay(model, buffer))
    task.use(speed_profile.after)

    task.use(save_and_quit(task, buffer, model, 195, 3))

    task.run(max_step=10000)


if __name__ == "__main__":
    main()
