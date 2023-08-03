from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym


# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")

# reset the environment to get the first observation
done = False
observation, info = env.reset()
env.step(1)
env.render()

# # observation = (16, 9, False)
# plt.imshow(np.eye(5))
# plt.show()



