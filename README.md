# RL with Atari

## Install

First, install gym and atari environments. May need to install other dependencies depending on your system.

```
pip install gym
```

and then install atari with one of the following commands
```
pip install "gym[atari]"
pip install gym[atari]
```

Use a version greater than 1 for Tensorflow.


## Environment

### Pong-v0

- Play against a decent AI player.
- One player wins if the ball pass through the other player and gets reward +1 else -1.
- Episode is over when one of the player reaches 21 wins
- final score is between -21 or +21 (lost all or won all)

```python
# action = int in [0, 6)
# state  = (210, 160, 3) array
# reward = 0 during the game, 1 if we win, -1 else
```

Use a modified env where the dimension of the input is reduced to

```python
# state = (80, 80, 1)
```

with downsampling and greyscale.

## Training

First launch test run on the Test environment
```
python t2_linear.py
```
and
```
python t3_nature.py
```

Then launch the training of DeepMind's DQN on pong with

```
python t5_train_atari_nature.py
```

The default config file should be sufficient to reach good performance ~5 million steps.



Training tips: 
(1) The code writes summaries of a bunch of useful variables that can help to monitor the training process.
Monitor the training with Tensorboard by doing:

```
tensorboard --logdir=results
```

and then connect to `ip-of-machine:6006`


(2) In case 'ROM is missing' for your game, remember to download http://www.atarimania.com/roms/Roms.rar and extract the .rar file.
    Here folder: ./Roms with extracted files is already provided.
    To import ROMS, just run:
```
python -m atari_py.import_roms <folder path to the extracted .rar file>
```

(3) By default, t5 records an episode video with ffmpeg package, on Ubuntu you can install it as:
```
apt update
apt install ffmpeg
```
