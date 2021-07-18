import tensorflow as tf
import numpy as np
from tensorflow import keras 
from collections import deque

###                                              ###
### ----- This code does not work properly ----- ###
###                                              ###

# After a lot of work and debugging, we found out that the gradient are not computed and we haven't been able to figure out why

class DuellingDQN(tf.keras.Model):

    def __init__(self):
        super(DuellingDQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(231, activation=tf.nn.relu, input_shape = (231, ))
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        
        self.pre_v1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.pre_v2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.v = tf.keras.layers.Dense(1, activation=None)

        self.pre_a1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.pre_a2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.a = tf.keras.layers.Dense(5, activation=None)

    # In the call method we have to define the forward pass of the model
    
    def call(self, inputs):

        x = self.dense1(inputs)
        x = self.dense2(x)

        v = self.pre_v1(x)
        v = self.pre_v2(v)
        v = self.v(v)

        a = self.pre_a1(x)
        a = self.pre_a2(a)
        a = self.a(a)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))    
        return Q

class DQN(tf.keras.Model):

    def __init__(self):
        super(DQN, self).__init__()   
        self.dense1 = tf.keras.layers.Dense(231, activation=tf.nn.relu, input_shape = (231, ))
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    # In the call method we have to define the forward pass of the model
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x

class experience_replay:

    def __init__(self, buffer_size):
        self.buffer = deque(maxlen = buffer_size)    

    def add_experience(self, state, action, reward, next_state, done):
        exp = (state, action, reward, next_state, done)
        self.buffer.append(exp)

    def remove_batch(self, batch_size):
        batch_idx = np.random.choice(len(self.buffer), batch_size, replace = False)
        batch = np.array(self.buffer, dtype='object')[batch_idx.astype(int)]
        return batch

class agent():

    def __init__(self, learning_rate, gamma, batch_size):

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = 1.
        self.min_eps = 0.001
        self.batch_size = batch_size
        self.buffer_len = 10000
        self.replay = experience_replay(self.buffer_len)

        self.model = DuellingDQN()
        self.model_target = DuellingDQN()   

        # The casting of the optimizer's parameters into tf.Variables is necessary to solve a problem when loading the checkpoint
        self.adam = tf.keras.optimizers.Adam(
            learning_rate = tf.Variable(self.learning_rate),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
            # clipnorm=10
        )
        self.adam.iterations  
        self.adam.decay = tf.Variable(0.0)

        self.model.compile(
            optimizer = self.adam,
            loss = tf.keras.losses.MeanSquaredError()
        )

        # This network Q' is not trained, periodically we'll copy the weights of the Q net into it
        self.model_target.compile(
            optimizer = self.adam,
            loss = tf.keras.losses.MeanSquaredError()
        )

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.adam)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory='.vscode/DLproject/checkpoints/', max_to_keep=5)
        

    def act(self, state):

        if np.random.rand() <= self.eps:
            action = np.random.choice(range(0, 5))
            return action
        
        else:
            action = np.argmax(self.model(np.array(state)))
            return action

    def train(self):

        # If the replay buffer doesn't reach the dimension of the desired batch, we continue to accumulate experiences
        if len(self.replay.buffer) < self.batch_size:
            return

        # Load a batch from the experience replay buffer and unpack it
        batch = self.replay.remove_batch(self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        target_q = []
        for exp in range(self.batch_size):
            
            state, action, reward, next_state, done = batch[exp]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        states = np.array(states).reshape(self.batch_size, 231)
        actions = np.array(actions).reshape(self.batch_size, 1)
        rewards = np.array(rewards).reshape(self.batch_size, 1)
        dones = np.array(dones).reshape(self.batch_size, 1)
        next_states = np.array(next_states).reshape(self.batch_size, 231)

        q_val_target = self.model_target.predict(np.array(next_states).reshape(self.batch_size, 231))  
        max_q = q_val_target.max(axis=1).reshape(self.batch_size, 1)
        q_val_model = self.model.predict(np.array(states).reshape(self.batch_size, 231))

        for i, row in enumerate(q_val_model):
            row[actions[i]] = rewards[i] + self.gamma * max_q[i] * (1 - dones[i])
        
        self.model.fit(states, q_val_model, batch_size = self.batch_size, verbose=0)     # Runs a single gradient update on a single batch of data
        

        """
        mse = tf.keras.losses.MeanSquaredError()
        loss = tf.keras.losses.mse(Q_targets, Q_predictions)
        var_list_fn = lambda: self.model.trainable_weights

        loss_fn = lambda: tf.keras.losses.mse(Q_predictions, Q_targets)
        loss_fn = tf.keras.losses.MeanSquaredError()
        var_list_fn = self.model.trainable_variables    # list object containing Tensors

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            pred = self.model(tf.Variable(np.array(states).reshape(self.batch_size, 231)), training=True)
            loss = loss_fn(Q_targets, Q_predictions)
            print(loss)
        gradient = tape.gradient(loss, q_val_model)
        print('tape grad {}'.format(gradient))
        # opt = self.adam.minimize(loss_fn, var_list=var_list_fn)
        """
       
    def copy_weights(self):
        self.model_target.set_weights(self.model.get_weights())

    def update_eps(self):
        if self.eps > self.min_eps:
            self.eps = self.eps * 0.995

    def save_model(self):
        self.manager.save()
        return

    def load_model(self):
        self.model.predict(np.zeros([1,231]))
        self.model_target.predict(np.zeros([1,231]))
        status = self.checkpoint.restore(self.manager.latest_checkpoint).assert_consumed()
        return