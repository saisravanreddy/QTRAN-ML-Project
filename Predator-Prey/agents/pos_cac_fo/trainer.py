#!/usr/bin/env python
# coding=utf8

"""
===========================================
 :mod:`qlearn` Q-Learning
===========================================


설명
=====

Choose action based on q-learning algorithm
"""

import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from agents.pos_cac_fo.agent import Agent
from agents.simple_agent import RandomAgent as NonLearningAgent
from agents.evaluation import Evaluation
from agents.simple_agent import StaticAgent as StAgent
from agents.simple_agent import ActiveAgent as AcAgent
import logging
import config
from envs.gui import canvas
import time

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 2.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Centralized DQN Trainer is created")

        self._env = env 
        self._eval = Evaluation()
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._agent_profile = self._env[0].get_agent_profile()
        self._agent_precedence = self._env[0].agent_precedence

        self._agent = Agent(self._agent_profile["predator"]["act_dim"], self._agent_profile["predator"]["obs_dim"][0])
        self._prey_agent = AcAgent(5)

        self.epsilon = 1.0

        if FLAGS.load_nn:
            self.epsilon = epsilon_min

        if FLAGS.gui:
            self.canvas = canvas.Canvas(self._n_predator, self._n_prey, FLAGS.map_size)
            self.canvas.setup()
    def learn_parallel(self):

        step = 0
        episode = 1 #episode to start
        print_flag = True
        count = 1
        while step < training_step:
            # parallelizing episodes start
            start_time = time.clock()

            total_reward = np.zeros(FLAGS.parallel_episodes)
            total_reward_pos = np.zeros(FLAGS.parallel_episodes)
            total_reward_neg = np.zeros(FLAGS.parallel_episodes)
            obs_parallel = [None]*FLAGS.parallel_episodes
            state_parallel = [None]*FLAGS.parallel_episodes
            episode_done_parallel = np.zeros(FLAGS.parallel_episodes,dtype=bool)

            episode_start = episode
            ep_step = 1 #episode step to start
            parallel_episode_no = 1
            while parallel_episode_no <=FLAGS.parallel_episodes:
                obs = self._env[parallel_episode_no-1].reset()
                obs_parallel[parallel_episode_no-1]=obs
                #obs_parallel.append(obs)
                state = self._env[parallel_episode_no-1].get_full_encoding()[:, :, 2]
                state_parallel[parallel_episode_no-1]= state
                #state_parallel.append(state)
                episode += 1
                parallel_episode_no += 1


            while ep_step <= FLAGS.max_step:
                #print "episode_start" , episode_start, "ep_step", ep_step
                episode = episode_start
                parallel_episode_no = 1

                #self._env[parallel_episode_no-1].render()

                #print "getting action"
                action_parallel = self.get_action_parallel(obs_parallel, step, state_parallel)
                #print "got action"
                obs_n_parallel = [None] * FLAGS.parallel_episodes
                state_n_parallel = [None] * FLAGS.parallel_episodes
                reward_parallel = [None]*(FLAGS.parallel_episodes)
                done_single_parallel = [None]*(FLAGS.parallel_episodes)

                while parallel_episode_no <=FLAGS.parallel_episodes:
                    if episode_done_parallel[parallel_episode_no-1]==False:
                        obs_n, reward, done, info = self._env[parallel_episode_no-1].step(action_parallel[parallel_episode_no-1])
                        obs_n_parallel[parallel_episode_no-1]=obs_n
                        #obs_n_parallel.append(obs_n)
                        reward_parallel[parallel_episode_no-1]=reward
                        #reward_parallel.append(reward)
                        state_n = self._env[parallel_episode_no-1].get_full_encoding()[:, :, 2]
                        state_n_parallel[parallel_episode_no-1]=state_n
                        #state_n_parallel.append(state_n)
                        done_single = sum(done) >0
                        done_single_parallel[parallel_episode_no-1]=(done_single)
                        self.train_agents(state_parallel[parallel_episode_no-1],
                                          action_parallel[parallel_episode_no-1],
                                          reward, state_n, done_single)

                        total_reward[parallel_episode_no-1] += np.sum(reward)
                        if np.sum(reward) >= 0:
                            total_reward_pos[parallel_episode_no-1] += np.sum(reward)
                        else:
                            total_reward_neg[parallel_episode_no-1] += np.sum(reward)
                        if is_episode_done(done, step) or ep_step >= FLAGS.max_step:
                            # print step, ep_step, total_reward
                            if print_flag and episode % FLAGS.eval_step == 1:
                                print "[train_ep %d]" % (episode), "\treward", total_reward[parallel_episode_no-1] #total_reward_pos, total_reward_neg
                            #print "TRAINING TIME for ", episode, " no: training episode ", (
                            #    step), "done training steps  (sec)", time.clock() - start_time
                            episode_done_parallel[parallel_episode_no - 1] = is_episode_done(done, step)
                    episode += 1
                    parallel_episode_no += 1

                obs_parallel = obs_n_parallel
                state_parallel = state_n_parallel

                ep_step += 1
                step += FLAGS.parallel_episodes

                if episode % 1 == 0:
                    update_network_time = time.clock()
                    self._agent.update_network()
                    #print "update_network start time ", update_network_time, "total time", time.clock() - update_network_time

            episode = episode_start + FLAGS.parallel_episodes


            self.test_parallel(episode)


            # parallelizing episodes end
        self._eval.summarize()
        self._agent.writer.close()



    def learn(self):

        step = 0
        episode = 0 #episode to start
        print_flag = True
        count = 1
        while step < training_step:

            episode += 1
            ep_step = 0
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0
            self.random_action_generator()
            start_time = time.clock()
            while True:
                step += 1
                ep_step += 1
                get_action_time = time.clock()
                action = self.get_action(obs, step, state)
                #get_action takes 0.003 seconds for each step
                #print "get_action start time ",get_action_time, "total time", time.clock()-get_action_time
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]
                done_single = sum(done) > 0

                train_agents_time = time.clock()
                self.train_agents(state, action, reward, state_n, done_single)
                #print "train_agents start time ",train_agents_time, "total time", time.clock()-train_agents_time
                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)
                if np.sum(reward) >= 0:
                    total_reward_pos += np.sum(reward)
                else:
                    total_reward_neg += np.sum(reward)

                if is_episode_done(done, step) or ep_step >= FLAGS.max_step :
                    # print step, ep_step, total_reward
                    if print_flag and episode % FLAGS.eval_step == 1:
                        print "[train_ep %d]" % (episode), "\treward", total_reward_pos, total_reward_neg
                    print "TRAINING TIME for ", episode, " no: training episode ", (
                        step), "done training steps  (sec)", time.clock() - start_time
                    break


            #update network only once per 10 episodes
            if episode%1 ==0:
                update_network_time = time.clock()
                self._agent.update_network()
                print "update_network start time ", update_network_time, "total time", time.clock() - update_network_time

            if episode % FLAGS.eval_step == 0:
                self.test(episode)


        self._eval.summarize()
        self._agent.writer.close()



    def random_action_generator(self):
        rand_unit = np.random.uniform(size = (FLAGS.n_predator, 5))
        self.rand = rand_unit / np.sum(rand_unit, axis=1, keepdims=True)
        

    def get_action(self, obs, step, state, train=True):
        act_n = []
        if train == True:
            self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator

        action_list = self._agent.act(state)
        for i in range(self._n_predator):
            #choose a random action if step < FLAGS.m_size * FLAGS.pre_train_step and epsilon greedy
            if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
                action = np.random.choice(5)
                act_n.append(action)
            else:              
                act_n.append(action_list[i])



        # Action of prey
        for i in range(FLAGS.n_prey):
            act_n.append(self._prey_agent.act(state, i))
        # act_n[1] = 2

        return np.array(act_n, dtype=np.int32)

    def get_action_parallel(self, obs_parallel, step, state_parallel, train=True):
        act_n = []
        if train == True:
            self.epsilon = max(self.epsilon - ((epsilon_dec)*FLAGS.parallel_episodes), epsilon_min)

        # Action of predator
        s_parallel = [None]*FLAGS.parallel_episodes
        for sravan in range(FLAGS.parallel_episodes):
            s = self._agent.state_to_index(state_parallel[sravan])
            s_parallel[sravan]=s

        action_parallel = self._agent.sess.run(self._agent.q_network.actor_network,feed_dict={self._agent.q_network.s_in: s_parallel})

        action_n_parallel = action_parallel.tolist()#[[None]*self._n_predator for i in range(FLAGS.parallel_episodes)]#[[None]*self._n_predator]*FLAGS.parallel_episodes
        for parallel_episode in range(FLAGS.parallel_episodes):
            for predator in range(self._n_predator):
                #choose a random action if step < FLAGS.m_size * FLAGS.pre_train_step and epsilon greedy
                if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
                    action = np.random.choice(5)
                    #act_n.append(action)

                    action_n_parallel[parallel_episode][predator] = action
                #else:
                #    if train:
                #        print "taking correct non random action"

                #else:
                #    act_n.append(action_list[i])
            for i in range(FLAGS.n_prey):
                (action_n_parallel[parallel_episode]).append(self._prey_agent.act(state_parallel[parallel_episode], i))

        return action_n_parallel

        #stp = self.sess.run(self.tp, feed_dict={self.s_in: [data[3] for data in minibatch]})
        '''
        action_list = self._agent.act(state)

        def act(self, state):
    
            predator_rand = np.random.permutation(FLAGS.n_predator)
            prey_rand = np.random.permutation(FLAGS.n_prey)      
    
            s = self.state_to_index(state)
        
            action = self.q_network.get_action(s[None])[0]

            def get_action(self, state_ph):
                #return self.sess.run(self.cpu_actor_network, feed_dict={self.s_in: state_ph})
                #sravan
                return self.sess.run(self.actor_network, feed_dict={self.s_in: state_ph})
        
        
            return action
        
        

        for i in range(self._n_predator):
            #choose a random action if step < FLAGS.m_size * FLAGS.pre_train_step and epsilon greedy
            if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
                action = np.random.choice(5)
                act_n.append(action)
            else:
                act_n.append(action_list[i])
        


        # Action of prey
        for i in range(FLAGS.n_prey):
            act_n.append(self._prey_agent.act(state, i))
        # act_n[1] = 2
        

        return np.array(act_n, dtype=np.int32)
        '''

    def train_agents(self, state, action, reward, state_n, done):
        self._agent.train(state, action, reward, state_n, done)

    def test_parallel(self, curr_ep=None):
        step = 0
        episode = 1

        test_flag = FLAGS.kt
        sum_reward = 0
        sum_reward_pos = 0
        sum_reward_neg = 0
        while step < testing_step:
            #parallelizing start
            start_time = time.clock()

            total_reward = np.zeros(FLAGS.parallel_episodes)
            total_reward_pos = np.zeros(FLAGS.parallel_episodes)
            total_reward_neg = np.zeros(FLAGS.parallel_episodes)

            obs_parallel = [None]*FLAGS.parallel_episodes
            state_parallel = [None]*FLAGS.parallel_episodes
            episode_done_parallel = np.zeros(FLAGS.parallel_episodes,dtype=bool)


            episode_start = episode
            ep_step = 1 #episode step to start
            parallel_episode_no = 1

            while parallel_episode_no <=FLAGS.parallel_episodes:
                obs = self._env[parallel_episode_no-1].reset()
                obs_parallel[parallel_episode_no-1]=obs
                #obs_parallel.append(obs)
                state = self._env[parallel_episode_no-1].get_full_encoding()[:, :, 2]
                state_parallel[parallel_episode_no-1]= state
                #state_parallel.append(state)
                episode += 1
                parallel_episode_no += 1

            while ep_step <= FLAGS.max_step:
                #print "testing episode_start" , episode_start, "testing ep_step", ep_step

                episode = episode_start
                parallel_episode_no = 1

                #print "getting test action"
                action_parallel = self.get_action_parallel(obs_parallel, step, state_parallel,False)
                #print "got test action"
                obs_n_parallel = [None] * FLAGS.parallel_episodes
                state_n_parallel = [None] * FLAGS.parallel_episodes
                reward_parallel = [None]*(FLAGS.parallel_episodes)
                done_single_parallel = [None]*(FLAGS.parallel_episodes)

                while parallel_episode_no <=FLAGS.parallel_episodes:
                    if episode_done_parallel[parallel_episode_no-1]==False:
                        obs_n, reward, done, info = self._env[parallel_episode_no-1].step(action_parallel[parallel_episode_no-1])
                        obs_n_parallel[parallel_episode_no-1]=obs_n
                        #obs_n_parallel.append(obs_n)
                        reward_parallel[parallel_episode_no-1]=reward
                        #reward_parallel.append(reward)
                        state_n = self._env[parallel_episode_no-1].get_full_encoding()[:, :, 2]
                        state_n_parallel[parallel_episode_no-1]=state_n
                        #state_n_parallel.append(state_n)
                        done_single = sum(done) >0
                        done_single_parallel[parallel_episode_no-1]=(done_single)

                        '''parallelize canvas here
                        state_next = state_to_index(state_n)
                        if FLAGS.gui:
                            self.canvas.draw(state_next, done, "Score:" + str(total_reward) + ", Step:" + str(ep_step))
                        '''

                        total_reward[parallel_episode_no-1] += np.sum(reward)
                        if np.sum(reward) >= 0:
                            total_reward_pos[parallel_episode_no-1] += np.sum(reward)
                        else:
                            total_reward_neg[parallel_episode_no-1] += np.sum(reward)
                        if is_episode_done(done, step,"test") or ep_step >= FLAGS.max_step:
                            # print step, ep_step, total_reward
                            #print "TESTING TIME for ", episode, "no: test episode ", step, "done testing steps (sec)", time.clock() - start_time

                            '''parallelize canvas
                            if FLAGS.gui:
                                self.canvas.draw(state_next, done, "Hello",
                                                 "Score:" + str(total_reward) + ", Step:" + str(ep_step))
                            '''

                            episode_done_parallel[parallel_episode_no - 1] = is_episode_done(done, step)
                    episode += 1
                    parallel_episode_no += 1

                obs_parallel = obs_n_parallel
                state_parallel = state_n_parallel

                ep_step += 1
                step += FLAGS.parallel_episodes

            episode = episode_start + FLAGS.parallel_episodes

            sum_reward += np.sum(total_reward)
            sum_reward_pos += np.sum(total_reward_pos)
            sum_reward_neg += np.sum(total_reward_neg)


            #parallelizing end


        if FLAGS.scenario == "pursuit":
            print "Test result: Average steps to capture: ", curr_ep, float(step) / episode
            self._eval.update_value("training result: ", float(step) / episode, curr_ep)
        elif FLAGS.scenario == "endless" or FLAGS.scenario == "endless2" or FLAGS.scenario == "endless3":
            print "Average reward:", FLAGS.penalty, curr_ep, sum_reward / episode, sum_reward_pos / episode, sum_reward_neg / episode
            self._eval.update_value("training result: ", sum_reward / episode, curr_ep)
            self._agent.logging(sum_reward/episode, curr_ep * 100)

    def test(self, curr_ep=None):

        step = 0
        episode = 0

        test_flag = FLAGS.kt
        sum_reward = 0
        sum_reward_pos = 0
        sum_reward_neg = 0
        while step < testing_step:
            episode += 1
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            if test_flag:
                print "\nInit\n", state
            total_reward = 0
            total_reward_pos = 0
            total_reward_neg = 0

            ep_step = 0
            start_time = time.clock()

            while True:

                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state, False)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]
                state_next = state_to_index(state_n)
                if FLAGS.gui:
                    self.canvas.draw(state_next, done, "Score:" + str(total_reward) + ", Step:" + str(ep_step))

                if test_flag:
                    aa = raw_input('>')
                    if aa == 'c':
                        test_flag = False
                    print action
                    print state_n
                    print reward

                obs = obs_n
                state = state_n
                r = np.sum(reward)
                # if r == 0.1:
                #     r = r * (-1.) * FLAGS.penalty
                total_reward += r # * (FLAGS.df ** (ep_step-1))
                if r > 0:
                    total_reward_pos += r
                else:
                    total_reward_neg -= r


                if is_episode_done(done, step, "test") or ep_step >= FLAGS.max_step:
                    print "TESTING TIME for ", episode , "no: test episode " , step , "done testing steps (sec)", time.clock() - start_time

                    if FLAGS.gui:
                        self.canvas.draw(state_next, done, "Hello", "Score:" + str(total_reward) + ", Step:" + str(ep_step))

                    break
            sum_reward += total_reward
            sum_reward_pos += total_reward_pos
            sum_reward_neg += total_reward_neg

        if FLAGS.scenario =="pursuit":
            print "Test result: Average steps to capture: ", curr_ep, float(step)/episode
            self._eval.update_value("training result: ", float(step)/episode, curr_ep)
        elif FLAGS.scenario =="endless" or FLAGS.scenario =="endless2" or FLAGS.scenario =="endless3":
            print "Average reward:", FLAGS.penalty, curr_ep, sum_reward /episode, sum_reward_pos/episode, sum_reward_neg/episode
            self._eval.update_value("training result: ", sum_reward/episode, curr_ep)
            self._agent.logging(sum_reward/episode, curr_ep * 100)


def is_episode_done(done, step, e_type="train"):

    if e_type == "test":
        if sum(done) > 0 or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if sum(done) > 0 or step >= FLAGS.training_step:
            return True
        else:
            return False

def state_to_index(state):
    """
    For the single agent case, the state is only related to the position of agent 1
    :param state:
    :return:
    """

    ret = np.zeros(2 * (FLAGS.n_predator + FLAGS.n_prey))
    for i in range(FLAGS.n_predator + FLAGS.n_prey):
        p = np.argwhere(np.array(state)==i+1)[0]
        #p = self.get_pos_by_id(state, i+1)
        ret[2 * i] = p[0]
        ret[2 * i + 1] = p[1]


    return ret

    


