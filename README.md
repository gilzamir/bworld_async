One Step DQN Asyncronous Implementation
=

This repo contains the Asyncronous Advantage Actor-Critic (A3C) algorithm implementation. The key features of this implementation is:

* The use of multiprocessing (indeed threading module) for real advantage of multi-core processors. 
* Use keras hight level features with minimum backend exposition. But it yet depends of tensorflow backend.

The code available contains experiments and test of prove code based em OpenGym Atari 2600 environments, but future extensions includes my own environment: a tridimentional game like environment. The goal this repo is to provide the community with documented example of A3C algorithm based in multiprocessing python library.

Training
=

Run the follow commands in system terminal:

$python3 learning_async.py

Contact
---
>For more questions, send me mail: gilzamir@gmail.com.
