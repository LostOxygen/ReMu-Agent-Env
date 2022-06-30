import gym
from gym.envs.registration import register

from stable_baselines3 import DQN

from ai_wars.constants import MAX_ITERATIONS

register(
	id="ai_wars/race-v0",
	entry_point="ai_wars.dqn_stable_baseline.dqn_env:ClientEnvironment",
	reward_threshold=10000
)

env = gym.make("ai_wars/race-v0", name="Peter")

model = DQN("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=MAX_ITERATIONS)

last_gamestate = env.reset()
while True:
	action, _ = model.predict(last_gamestate)
	last_gamestate, _, _, _ = env.step(action)
