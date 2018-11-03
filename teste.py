
# Import the gym module

import gym



# Create a breakout environment

env = gym.make('BreakoutDeterministic-v4')

i = 0

while i < 100:
  # Reset it, returns the starting frame

  frame = env.reset()

  # Render

  env.render()



  is_done = False

  LIVES = 5

  while not is_done:

    # Perform a random action, returns the new frame, reward and whether the game is over

    frame, reward, is_done, info = env.step(3)
    # Render
    if info['ale.lives'] == 1:
      print('teste')

    env.render()
  i += 1