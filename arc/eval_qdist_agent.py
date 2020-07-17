from arc.grad_descent_render import QDistanceDescentAgent
from multi_goal.GenerativeGoalLearning import evaluate
from multi_goal.agents import HERSACAgent
from multi_goal.envs.toy_labyrinth_env import ToyLab

seed = 3
env = ToyLab(seed=seed)
hersac = HERSACAgent(env=env, rank=seed)
agent = QDistanceDescentAgent(hersac=hersac)
evaluate(agent=agent, env=env, very_granular=False)
input("exit")
