import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from t1_schedule import LinearExploration, LinearSchedule

from configs.t2_linear import config


class Linear(DQN):
    """
    Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        """
        Add placeholders:
            Note stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32
        N.B: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            Can also use the state_shape computed above.
        """
        ih = state_shape[0]               # state image height
        iw = state_shape[1]               # state image width
        ic = state_shape[2]               # RGB channels of 3
        fs = self.config.state_history    # stack of 4 frames

        self.s = tf.placeholder(tf.uint8, shape=(None, ih, iw, ic*fs), name="states")
        self.a = tf.placeholder(tf.int32, shape=(None), name="actions")
        self.r = tf.placeholder(tf.float32, shape=(None), name="rewards")
        self.sp = tf.placeholder(tf.uint8, shape=(None, ih, iw, ic*fs), name="n_states")
        self.done_mask = tf.placeholder(tf.bool, shape=(None), name="isdone")
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")



    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        """
            a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.
            -make sure to specify the scope and reuse
        """
        with tf.variable_scope(scope, reuse=reuse):
            inp = tf.layers.flatten(state)
            out = tf.layers.dense(inp, num_actions, use_bias=True)

        return out



    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Note that in DQN, it maintains two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope
        mechanism in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we should update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        N.B: 
            Add an operator self.update_target_op that for each variable in
            tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

          - tf.group (the * operator can be used to unpack a list)
          - be sure to set self.update_target_op
        """
        # with tf.variable_scope(q_scope):          # incorrect use #
        src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        # with tf.variable_scope(target_q_scope):   # incorrect use #
        dst_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)

        update = [ 
            tf.assign(dst_vars[i], src_vars[i])
            for i in range(len(src_vars))
        ]
        self.update_target_op = tf.group(*update)



    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        """
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
         N.B: 
            - Config variables are accessible through self.config
            - can access placeholders like self.a (for actions)
               self.r (rewards) or self.done_mask for instance
            - following functions are useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        q_samp = tf.where(self.done_mask, self.r, self.r + self.config.gamma * tf.reduce_max(target_q, axis=1))
        a_vec = tf.one_hot(self.a, num_actions)
        q_s_a = tf.reduce_sum(tf.multiply(q, a_vec), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_s_a))



    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """
        ##############################################################
        """
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

          N.B: following functions are useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
          we can access config variables by writing self.config.variable_name
        """
        adam = tf.train.AdamOptimizer(learning_rate=self.lr)
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        gradients  = adam.compute_gradients(self.loss, var_list=trainables)
        if self.config.grad_clip:
            gradients = [
                (tf.clip_by_norm(grad_v[0], self.config.clip_val), grad_v[1])
                for grad_v in gradients
            ]
        self.train_op = adam.apply_gradients(gradients)
        self.grad_norm = tf.global_norm([v[0] for v in gradients])
        


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
