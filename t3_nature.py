import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from t1_schedule import LinearExploration, LinearSchedule
from t2_linear import Linear


from configs.t3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        The computation of Q values like in the paper
            https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

            the section "model architecture" of the appendix of the nature paper particulary useful.
            store result in out of shape = (batch_size, num_actions)
        N.B: 
            - following functions are useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Need to specify the scope and reuse
        """
        with tf.variable_scope(scope, reuse=reuse):
            conv1 = tf.layers.conv2d(state, filters=32, kernel_size=8, strides=4, padding="same", activation=tf.nn.relu, name="conv1")
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu, name="conv2")
            conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, name="conv3")
            fc_in = tf.layers.flatten(conv3)
            aff_1 = tf.layers.dense(fc_in, units=512, activation=tf.nn.relu, name="affine1")
            out   = tf.layers.dense(aff_1, units=num_actions, activation=None, name="output")

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
