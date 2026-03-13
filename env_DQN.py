import collections
import numpy as np
import torch
import pandas as pd
import random
import torch.nn as nn
import math
from dataset_processing import BasicBuffer,cach_hit_ratio,cach_hit_ratio2
from model_ddpm import DuelingDQN


class CacheEnv(object):

    def __init__(self, popular_content,cache_size):
        self.cache_size=cache_size
        self.popular_content = popular_content       # 推荐的电影
        self.action_bound = [0,1]
        self.reward = np.zeros(shape=1, dtype=float)

        #cache list
        if len(self.popular_content) < self.cache_size:
            self.state = self.popular_content
        # 大于情况
        if len(self.popular_content) >= self.cache_size:
            # self.state = random.sample(list(self.popular_content), self.cache_size) # 状态是随机采样的100个推荐电影
            self.state = list(self.popular_content)[:self.cache_size]
        # state1 = []
        # for i in range(len(self.popular_content)):
        #     # 按照内容流行度进行排序
        #     if self.popular_content[i] in self.state:
        #         state1.append(self.popular_content[i])
        # self.state = state1

        self.last_content=[]        # 剩下的电影
        for i in range(len(self.popular_content)):
            if self.popular_content[i] not in self.state:
                self.last_content.append(self.popular_content[i])
        # print('self.last_content',len(self.last_content))
        # print('self.cache_size',self.cache_size)

        # RSU2 缓存
        # 剩余电影小于缓存容量（不需要考虑）
        if len(self.last_content)<self.cache_size:
            self.state2 = []
            for i in range(len(self.last_content)):
                if self.last_content[i] not in self.state:
                    self.state2.append(self.last_content[i])

        # 剩余电影大于等于缓存容量
        if len(self.last_content)>=self.cache_size:
            self.state2 = random.sample(list(self.last_content), self.cache_size)

        self.last_content2 = []  # 剩下的电影
        for i in range(len(self.last_content)):
            if self.last_content[i] not in self.state2:
                self.last_content2.append(self.last_content[i])

        # RSU3 缓存
        if len(self.last_content2)>=self.cache_size:
            self.state3 = random.sample(list(self.last_content2), self.cache_size)

        self.last_content3 = []  # 剩下的电影
        for i in range(len(self.last_content2)):
            if self.last_content2[i] not in self.state3:
                self.last_content3.append(self.last_content2[i])

        # RSU4 缓存
        if len(self.last_content3)>=self.cache_size:
            self.state4 = random.sample(list(self.last_content3), self.cache_size)

        self.last_content4 = []  # 剩下的电影
        for i in range(len(self.last_content3)):
            if self.last_content3[i] not in self.state4:
                self.last_content4.append(self.last_content3[i])



        # 2个RSU分别存放推荐的电影,第一个放前100个，后一个放后100个
        self.init_state = self.state.copy()
        self.init_cash2 = self.state2.copy()
        self.init_cash3 = self.state3.copy()
        self.init_cash4 = self.state4.copy()
        self.init_last_content = self.last_content.copy()

    def step(self, action, test_dataset, v2i_rate,v2i_rate_mbs, vehicle_epoch, vehicle_request_num, print_step):
        action = np.clip(action, *self.action_bound)

        if action == 1:

            if len((self.last_content))>=5:
                replace_content = random.sample(list(self.last_content), 5)
                count = 0
                if count < 5:
                    self.state[-count - 1] = replace_content[count]
                    count += 1
            else:
                replace_content = self.last_content
            # count = 0
            # if count < 5:
            #     self.state[-count-1]=replace_content[count]
            #     count+=1

            state1 = []
            for i in range(len(self.popular_content)):
                # 按照内容流行度进行排序
                if self.popular_content[i] in self.state:
                    state1.append(self.popular_content[i])
            self.state = state1
            # 剩余未缓存内容
            last_content=[]
            for i in range(len(self.popular_content)):
                if self.popular_content[i] not in self.state:
                    last_content.append(self.popular_content[i])
            self.last_content=last_content
            # print('self.last_content:', len(self.last_content))

            #剩余内容排序
            last_content_new = []
            for i in range(len(self.popular_content)):
                # 按照内容流行度进行排序
                if self.popular_content[i] in self.last_content:
                    last_content_new.append(self.popular_content[i])
            self.last_content = last_content_new

            # 小于缓存容量（不考虑）
            if len(self.last_content)<=self.cache_size:
                self.state2 = self.last_content

            # 大于缓存容量，RSU2 缓存替换
            if len(self.last_content)>self.cache_size:
                # self.state2 = random.sample(list(self.last_content), self.cache_size)
                self.state2 = list(self.last_content)[:self.cache_size]


            # 剩下的电影
            self.last_content2 = []  # 剩下的电影
            for i in range(len(self.last_content)):
                if self.last_content[i] not in self.state2:
                    self.last_content2.append(self.last_content[i])

            # RSU3 缓存
            if len(self.last_content2) >= self.cache_size:
                # self.state3 = random.sample(list(self.last_content2), self.cache_size)
                self.state3 = list(self.last_content2)[:self.cache_size]
            # 剩下的电影
            self.last_content3 = []  # 剩下的电影
            for i in range(len(self.last_content2)):
                if self.last_content2[i] not in self.state3:
                    self.last_content3.append(self.last_content2[i])

            # RSU4 缓存
            if len(self.last_content3) >= self.cache_size:
                # self.state4 = random.sample(list(self.last_content3), self.cache_size)
                self.state4 = list(self.last_content3)[:self.cache_size]
            # 剩下的电影
            self.last_content4 = []
            for i in range(len(self.last_content3)):
                if self.last_content3[i] not in self.state4:
                    self.last_content4.append(self.last_content3[i])





        all_vehicle_request_num = 0
        for i in range(len(vehicle_epoch)):
            all_vehicle_request_num += vehicle_request_num
        #print('=================================all_vehicle_request_num', all_vehicle_request_num,'================================')
        # print('RSU1 state内容：',self.state)
        # print('RSU2 state内容：', self.state2)
        cache_efficiency = cach_hit_ratio(test_dataset, self.state,
                                           all_vehicle_request_num)
        cache_efficiency2 = cach_hit_ratio2(test_dataset, self.state2 , self.state,
                                           all_vehicle_request_num)
        cache_efficiency3 = cach_hit_ratio(test_dataset, self.state3,
                                            all_vehicle_request_num)
        # print('len(self.state3):',len(self.state3))
        cache_efficiency4 = cach_hit_ratio(test_dataset, self.state4,
                                            all_vehicle_request_num)

        # print('cache_efficiency：', cache_efficiency)
        # print('cache_efficiency2：', cache_efficiency2)
        cache_efficiency = float(cache_efficiency)/100
        cache_efficiency2 = float(cache_efficiency2)/100
        cache_efficiency3 = float(cache_efficiency3) / 100
        cache_efficiency4 = float(cache_efficiency4) / 100

        reward=0
        request_delay=0
        for i in range(len(vehicle_epoch)):
            vehicle_idx=vehicle_epoch[i]
            reward += cache_efficiency * math.exp(-0.0001 * 8000000 / v2i_rate) * vehicle_request_num
            reward += cache_efficiency2 * math.exp(-0.0001 * 8000000 / v2i_rate
                                                    -0.4 * 8000000 / 15000000) * vehicle_request_num
            reward += cache_efficiency3 * math.exp(-0.0001 * 8000000 / v2i_rate
                                                   - 0.4 * 8000000 / 15000000) * vehicle_request_num
            reward += cache_efficiency4 * math.exp(-0.0001 * 8000000 / v2i_rate
                                                   - 0.4 * 8000000 / 15000000) * vehicle_request_num
            reward += (1-cache_efficiency-cache_efficiency2-cache_efficiency3-cache_efficiency4)\
                                        * math.exp(- 0.5999 * 8000000 / (v2i_rate/2))* vehicle_request_num



            request_delay += cache_efficiency * vehicle_request_num / v2i_rate*800


            #print(i,'local rsu delay', vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]*100000)
            request_delay += cache_efficiency2 * (
                    vehicle_request_num / v2i_rate+vehicle_request_num / 150000000) *800
            request_delay += cache_efficiency3 * (
                    vehicle_request_num / v2i_rate + 2*vehicle_request_num / 150000000) * 800
            request_delay += cache_efficiency4 * (
                    vehicle_request_num / v2i_rate + 3*vehicle_request_num / 150000000) * 800




            request_delay +=(1-cache_efficiency-cache_efficiency2)*(vehicle_request_num / (v2i_rate/2))*800

        request_delay = request_delay/len(vehicle_epoch)*1000

        if print_step % 50 ==0:
            print("---------------------------------------------")
            print('all_vehicle_request_num', all_vehicle_request_num)
            print('step:{} RSU1 cache_efficiency:{}'.format(print_step,cache_efficiency*100))
            print('step:{} RSU2 cache_efficiency:{}'.format(print_step,cache_efficiency2*100))
            print('step:{} RSU3 cache_efficiency:{}'.format(print_step, cache_efficiency3 * 100))
            print('step:{} RSU4 cache_efficiency:{}'.format(print_step, cache_efficiency4 * 100))
            print('step',print_step,'request delay:%f' %(request_delay))
            print("---------------------------------------------")
        return self.state, reward, cache_efficiency, cache_efficiency2,cache_efficiency3,cache_efficiency4, request_delay

    def reset(self):
        return self.init_state, self.init_cash2, self.init_last_content


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size
                     ,request_dataset, v2i_rate, v2i_rate_mbs,vehicle_epoch, vehicle_request_num):

    episode_rewards = []

    cache_efficiency_list=[]
    cache_efficiency2_list = []
    cache_efficiency3_list = []
    cache_efficiency4_list = []

    request_delay_list=[]

    for episode in range(max_episodes):
        state , _ , _= env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, cache_efficiency, cache_efficiency2,cache_efficiency3, cache_efficiency4, request_delay = env.step(action, request_dataset, v2i_rate,v2i_rate_mbs, vehicle_epoch, vehicle_request_num, step)
            agent.replay_buffer.push(state, action, reward, next_state)
            episode_reward += reward


            if len(agent.replay_buffer) % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                cache_efficiency_list.append(cache_efficiency)
                cache_efficiency2_list.append(cache_efficiency2)
                cache_efficiency3_list.append(cache_efficiency3)
                cache_efficiency4_list.append(cache_efficiency4)
                request_delay_list.append(request_delay)
                break

            state = next_state
        if episode == max_episodes-1:
            cache_efficiency_all = cache_efficiency + cache_efficiency2 + cache_efficiency3 + cache_efficiency4

    return episode_rewards, cache_efficiency_list, request_delay_list,cache_efficiency_all


class DQNAgent:

    # buffer size？
    def __init__(self, env, c_s, learning_rate=0.01, gamma=0.99, buffer_size=10000):
        self.env = env
        self.c_s = c_s
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(self.c_s, 2).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.DoubleTensor(state).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        action_bound = [0, 1]
        if (np.random.randn() > eps):
            action = random.sample(action_bound, 1)
            action = action[0]
            return action

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.DoubleTensor(next_states).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update(self, batch_size):
        for i in range(50):
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)
            # print("update loss ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()