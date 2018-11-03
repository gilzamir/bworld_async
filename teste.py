
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


  env.step(1)
  is_done = False

  LIVES = 5

  while not is_done:

    # Perform a random action, returns the new frame, reward and whether the game is over

    frame, reward, is_done, info = env.step(0)
    # Render
    if info['ale.lives'] < LIVES:
      env.step(1)
      LIVES -= 1

    env.render()
  i += 1