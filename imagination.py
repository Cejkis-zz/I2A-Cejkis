import random
import time
import threading
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
import sokoban
import datetime

print(str(datetime.datetime.now()))

episode = 0

r_lock = threading.Lock()
r_sum = 0
r_done = 0
r_done2 = 0


# approximate policy and value using Neural Network
# actor -> state is input and probability of each action is output of network
# critic -> state is input and value of state is output of network
# actor and critic network share first hidden layer
def build_model(state_size, action_size, network):
    input = Input(shape=state_size)
    model = Conv2D(filters= network[0], kernel_size=(4,4), strides=(2,2), activation='relu', padding='same',
                    data_format='channels_first')(input)
    model = Conv2D(filters= network[1], kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                   data_format='channels_first')(model)
    conv = Flatten()(model)
    fc = Dense(network[2], activation='relu')(conv)
    policy = Dense(action_size, activation='softmax')(fc)
    value = Dense(1, activation='linear')(fc)

    actor = Model(inputs=input, outputs=policy)
    critic = Model(inputs=input, outputs=value)

    actor._make_predict_function()
    critic._make_predict_function()

    # actor.summary()
    # critic.summary()

    return actor, critic


class A3CAgent:
    def __init__(self, alr, clr, act_rho,crit_rho , ae, ce, network):
        # environment settings
        self.state_size = sokoban.STATE_SIZE
        self.action_size = 4

        self.discount_factor = 0.99

        # optimizer parameters
        self.actor_lr = alr
        self.critic_lr = clr
        self.threads = 16

        self.act_rho = act_rho
        self.crit_rho = crit_rho
        self.act_eps = ae
        self.crit_eps = ce

        self.network = network

        print("alr clr ar cr ae ce " + str(alr) + " "+ str(clr) + " "+ str(act_rho) + " " + str(crit_rho) + " " + str(ae) + " "+ str(ce) + " ")

        # create model for actor and critic network
        self.actor, self.critic = build_model(self.state_size, self.action_size, self.network)

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    def train(self):

        global episode

        global weights
        if weights != "":
            self.load_model(weights)

        agents = [Agent(self, self.action_size, self.state_size, [self.actor, self.critic], self.sess, self.optimizer,
                        self.discount_factor, self.network) for _ in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while episode < EPISODES:
            time.sleep(3)
            #self.save_model("./save_model/breakout_a3c")


    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01*entropy
        optimizer = RMSprop(lr=self.actor_lr, rho=self.act_rho, epsilon=self.act_eps)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=self.crit_rho, epsilon=self.crit_eps)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

# make agents(local) and start training
class Agent(threading.Thread):
    def __init__(self, a3c, action_size, state_size, model, sess, optimizer, discount_factor, network):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.a3c = a3c

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = build_model(self.state_size, self.action_size, network)

        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.t_max = 20
        self.t = 0

    # Thread interactive with environment
    def run(self):

        global weights
        if weights != "":
            self.a3c.load_model(weights)

        global episode, network, r_sum, r_done, r_done2

        env = sokoban.Sokoban()

        small = True

        while episode < EPISODES:
            done = False

            score = 0
            s_ = env.reset()

            while not done:

                s = np.float32([s_])

                future_values = [0,0,0,0]
                states = np.zeros((4,4,8,5))

                for i in [0,1,2,3]:

                    env.newImagination()

                    im_, r, d = env.doImaginaryAction(i)
                    future_values[i] += r
                    im = np.float32([im_])

                    if not d:
                        for j in range(3):

                            action, policy = self.get_action(np.float32(im))

                            im_, r, d = env.doImaginaryAction(action)

                            future_values[i] += r

                            im = np.float32([im_])

                            if d:
                                break

                    states[i] = im_

                    #print(future_values)

                values = self.critic.predict(states)
                values = np.reshape(values, len(values))
                future_values += values

                min = np.min(future_values)
                future_values = [x - min + 0.0001 for x in future_values]

                sum = np.sum(future_values)
                future_values /= sum

                #print(str(min) + " " + str(sum) + " " + str(future_values))
                #action = np.argmax(future_values)

                action = np.random.choice(self.action_size, p=future_values)

                #action, policy = self.get_action(s)

                s_, reward, done = env.step(action)

                score += reward
                reward = np.clip(reward, -1., 1.) # todo

                self.memory(s, action, reward)

                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_localmodel()
                    self.t = 0

                if done:
                    with r_lock:  # můj kód
                        episode += 1

                        r_sum += score

                        if reward > 0.5:
                            if small:
                                r_done += 1
                            else:
                                r_done2 += 1

                        if episode % 500 == 0:

                            avg_done = round(r_done / 250.0, 3)
                            avg_done2 = round(r_done2 / 250.0, 3)

                            print(str(datetime.datetime.now()) + " " + str(episode) +
                                  " %.3f" % round(r_sum / 500.0, 3) +
                                  " %.3f" % avg_done + " %.3f" % avg_done2)

                            r_sum = 0
                            r_done = 0
                            r_done2 = 0

                    if small:
                        small = False
                    else:
                        small = True

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1]))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_model(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states),*self.state_size))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, history):

        policy = self.local_actor.predict(history)[0]

        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


if __name__ == "__main__":

    network = [32,32,256]
    EPISODES = 700000

    r_lastScore = 0.0
    episode = 0
    weights = ""
    #weights = "weights" + str(network[0]) + str(network[1]) + str(network[2]) + "/" + str(episode) + " " + str(r_lastScore)

    global_agent = A3CAgent(4e-5, 4e-5, 0.9, 0.9, 1e-8, 1e-8, network)
    global_agent.train()
