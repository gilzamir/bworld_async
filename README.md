One Step DQN Asyncronous Implementation
=

This repo contains a one step deep A3C and Q-learning asyncronous implementations and code base for future research in asyncronous multi-agent implementation of deep reinforcement learning algorithms. The key features of this implementation is:

* The use of multiprocessing (indeed threading module) for real advantage of multi-core processors. 
* Use keras hight level features with minimum backend exposition. But it yet depends of tensorflow backend.

The code available contains differents experiments and test of prove code based em OpenGym Atari 2600 environments.


Training
=

Run the follow commands in system terminal:

$python3 learning_async.py

Contact
---
>For more questions, send me mail: gilzamir@gmail.com.
