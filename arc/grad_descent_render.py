from typing import Callable

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from multi_goal.GenerativeGoalLearning import trajectory, Agent
from multi_goal.agents import HERSACAgent
from multi_goal.envs import Observation
from multi_goal.envs.toy_labyrinth_env import ToyLab


def distance_contours(dist_fn: Callable, goal: np.ndarray):
    fig, ax = plt.subplots()

    wall = np.array([[-1, 0], [0.5, 0]])
    ax.plot(*wall.T, c="red")

    pc = ax.scatter(*goal, c="orange")

    space2d = np.mgrid[-1:1:30j, -1:1:30j]
    isolines = 10
    dists = np.array([dist_fn(s, goal) for s in space2d.reshape((2, -1)).T])
    ax.contour(*space2d, dists.reshape((30, 30)), isolines, cmap=viridis)

    fig.tight_layout()

    return fig, ax, pc


def dist(x, y):
    return np.linalg.norm(np.subtract(x, y))


def init_qdist(hersac: HERSACAgent):
    sac = hersac._model.model.policy_tf

    def qdist(achieved: np.ndarray, desired: np.ndarray):
        flat_obs = np.concatenate((achieved, desired))[np.newaxis]
        return np.min(sac.sess.run([sac.qf1, sac.qf2], {sac.obs_ph: flat_obs}))
    return qdist


def get_new_input_goal():
    return np.array([float(i) for i in input("Put in new goal. e.g. '1.0 -0.1'\n").strip().split()])


class QDistanceDescentAgent(Agent):
    def __init__(self, hersac: HERSACAgent):
        self._sac = sac = hersac._model.model.policy_tf
        with sac.qf1.graph.as_default():
            Q = tf.reduce_mean(tf.stack([sac.qf1, sac.qf2]))
            self._dQds = tf.gradients(Q, sac.obs_ph)[0]

    def __call__(self, obs: Observation) -> np.ndarray:
        flat_obs = [[*obs.achieved_goal, *obs.desired_goal]]
        direction = self._sac.sess.run(self._dQds, {self._sac.obs_ph: flat_obs})[0][:2]
        norming = max(1, np.linalg.norm(direction))
        return direction / norming


    def train(self, timesteps: int) -> None:
        pass

    def reset_momentum(self):
        pass


if __name__ == '__main__':
    plt.ion()
    goal = np.array([0, 0.5])
    env = ToyLab(use_random_starting_pos=True, max_episode_len=60)
    her_sac = HERSACAgent(env=env, rank=3)
    qdist = init_qdist(hersac=her_sac)
    fig, ax, pc_goal = distance_contours(dist_fn=qdist, goal=goal)

    agent = QDistanceDescentAgent(hersac=her_sac)
    while True:
        agent.reset_momentum()
        tr = trajectory(pi=agent, env=env, goal=goal, print_every=100)
        xs = np.array([step[3].achieved_goal for step in tr])

        lines, = ax.plot([], [], c="blue", alpha=0.8)
        h = ax.scatter([], [], c="blue")

        for idx, head in enumerate(xs):
            curve = xs[:idx+1]
            lines.set_data(*curve.T)
            h.set_offsets(head)
            plt.pause(0.05)
