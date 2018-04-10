
import time
import sys
import threading
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K

# global variables for A3C
import sokoban
import datetime

print(str(datetime.datetime.now()))

global episode
episode = 0
EPISODES = 700000

r_lock = threading.Lock()
r_sum = 0
r_done = 0


# approximate policy and value using Neural Network
# actor -> state is input and probability of each action is output of network
# critic -> state is input and value of state is output of network
# actor and critic network share first hidden layer
def build_model(state_size, action_size):
    input = Input(shape=state_size)
    model = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same',
                   data_format='channels_first')(input)
    model = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                     data_format='channels_first')(model)
    conv = Flatten()(model)
    fc = Dense(256, activation='relu')(conv)
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
    def __init__(self, action_size, alr, clr, act_rho,crit_rho , ae, ce):
        # environment settings
        self.state_size = (4,8,5)
        self.action_size = action_size

        self.discount_factor = 0.99

        # optimizer parameters
        self.actor_lr = alr
        self.critic_lr = clr
        self.threads = 16

        self.act_rho = act_rho
        self.crit_rho = crit_rho
        self.act_eps = ae
        self.crit_eps = ce

        print("alr clr ar cr ae ce" + str(alr) + " "+ str(clr) + " "+ str(act_rho) + " " + str(crit_rho) + " " + str(ae) + " "+ str(ce) + " ")

        # create model for actor and critic network
        self.actor, self.critic = build_model(self.state_size, self.action_size)

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    def train(self):

        global episode
        episode = 0

        # self.load_model("./save_model/breakout_a3c")
        agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic], self.sess, self.optimizer,
                        self.discount_factor, [self.summary_op, self.summary_placeholders,
                        self.update_ops, self.summary_writer]) for _ in range(self.threads)]

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
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = build_model(self.state_size, self.action_size)

        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

        self.avg_p_max = 0
        self.avg_loss = 0

        # t_max -> max batch size for training
        self.t_max = 20
        self.t = 0

    # Thread interactive with environment
    def run(self):
        # self.load_model('./save_model/breakout_a3c')
        global episode

        env = sokoban.Sokoban()

        step = 0

        while episode < EPISODES:
            done = False

            # 1 episode = 5 lives
            score = 0
            s_ = env.reset()

            # At start of episode, there is no preceding frame. So just copy initial states to make history

            act_his = []

            while not done:
                step += 1
                self.t += 1
                s = np.float32([s_])

                # get action for the current history and go one step in environment
                action, policy = self.get_action(s)
                # change action to real_action
                act_his += [action]
                s_, reward, done = env.step(action)
                # pre-process the observation --> history

                self.avg_p_max += np.amax(self.actor.predict(np.float32(s))) # todo s_ ???

                score += reward
                reward = np.clip(reward, -1., 1.) # todo

                # save the sample <s, a, r, s'> to the replay memory
                self.memory(s, action, reward)


                #
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_localmodel()
                    self.t = 0

                # if done, plot the score over episodes
                if done:
                    with r_lock:  # můj kód
                        episode += 1

                        global r_sum, r_done

                        r_sum += score

                        if(reward>=0.5):
                            r_done +=1

                        if episode % 500 == 0:
                            print(str(datetime.datetime.now()) + " " + str(episode) + " %.3f" % round(r_sum / 500.0, 3) + " %.3f" % round(r_done / 500.0, 3))
                            r_sum = 0
                            r_done = 0

                    #print("episode:", episode, "  score:", score, "  step:", step, " path:", act_his )

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

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

    # global_agent = A3CAgent(4, 0.05e-4, 0.05e-4, 0.99, 0.99, 0.004, 0.004) # error spadl na -700
    # global_agent.train()

    EPISODES = 70000
    global_agent = A3CAgent(4, 0.1e-4, 0.1e-4, 0.99, 0.99, 0.004, 0.004) # zhruba stejny
    global_agent.train()



    # global_agent = A3CAgent(4, 1.5e-4, 0.5e-4, 0.99, 0.99, 0.004, 0.004) ## vyloženě špatný
    # global_agent.train()

    # global_agent = A3CAgent(4, 0.25e-4, 0.5e-4, 0.99, 0.99, 0.004, 0.004) # po 10k spadlo
    # global_agent.train()
