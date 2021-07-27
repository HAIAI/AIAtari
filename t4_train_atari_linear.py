import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from t1_schedule import LinearExploration, LinearSchedule
from t2_linear import Linear

from configs.t4_train_atari_linear import config

"""
Use linear approximation for the Atari game.
May change the configurations (in the configs/ folder). 

Find the results, log and video recordings of this agent every 250k under
the corresponding file in the results folder. Better monitor the progress
of the training is to use Tensorboard. 

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.

"""

if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
