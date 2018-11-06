import threading
import time

class Model:
  def __init__(self):
    self.x = 0
    self.printed = False

def printador(model):
  old_x = model.x
  print("Printador: ")
  print(old_x)
  model.printed = True
  while True:
    while old_x == model.x:
      time.sleep(1)
    print("Printador: ")
    print(model.x)
    old_x = model.x
    model.printed = True
def modificador(model):
  while True:
    if model.printed:
      model.x += 1
      model.printed = False
    time.sleep(1)

m = Model()
t1 = threading.Thread(target=printador, args=(m, ))
t2 = threading.Thread(target=modificador, args=(m, ))
t1.start()
t2.start()
t1.join()
t2.join()



'''
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
  '''