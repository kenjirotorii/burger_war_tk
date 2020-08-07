# -*- coding: utf-8 -*-
'''
DQN implementation with pytorch
'''

import random
import numpy as np
from collections import namedtuple

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


'''
Transitionの各要素のtype
state <float>: ndarray 2D, shape=(1, num_states), e.g. array([[0.1, 0.3, 0.4, 0.5]])
action <int>: ndarray 2D, shape=(1, 1), e.g. array([[2]])
next_state <float>: same as state above
reward <float>: ndarray 2D, shape=(1, 1), e.g. array([[1.0]])
'''
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

GAMMA = 0.95

# 損失関数huberの定義
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class Memory:

    def __init__(self, capacity):
        self.capacity = capacity  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class Brain:

    def __init__(self, num_states, num_actions, memory_cap):

        self.num_actions = num_actions
        self.num_states = num_states
        self.memory = Memory(memory_cap)  # 経験を記憶するメモリオブジェクトを生成

        # ニューラルネットワークを構築
        self.model = Sequential([
            Dense(72, activation='relu', input_shape=(num_states,)),
            Dense(72, activation='relu'),
            Dense(num_actions, activation='softmax')
        ])
        
        # 最適化手法の設定
        self.optimizer = Adam(lr=0.0005)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.model.summary()

    def replay(self, batch_size):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # ------------------------------
        # 1. メモリサイズの確認
        # ------------------------------

        # メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < batch_size:
            print("memory size is smaller than batch size.")
            return

        # ------------------------------
        # 2. ミニバッチの作成
        # ------------------------------

        # メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(batch_size) # [Transition(state=array(...), action=array(...), ...), Transition(), ...]

        # 各変数をミニバッチに対応する形に変形
        batch = Transition(*zip(*transitions)) # Transition(state=(array(...), array(...), ...), action=(...), ...)

        # 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        state_batch = np.concatenate(batch.state) # array([[0.1, 0.2, ....], [...], ...]), shape=(batch_size, num_states)
        action_batch = np.concatenate(batch.action) # array([[1], [2], ....]), shape=(batch_size, 1)
        reward_batch = np.concatenate(batch.reward) # array([[1.0], [0.0], ....]), shape=(batch_size, 1)
        
        # 最後のステップではnext_statesは存在しない(None)ので、最後のステップがバッチに含まれている場合にはそれを取り除く
        non_final_next_states = np.concatenate([s for s in batch.next_state if s is not None])

        # ------------------------------
        # 3. 教師データとなるQ(s,a)の値を求める
        # expected_state_action_valuesは、1step前のモデルと次の状態(next_states)を元に計算した教師データ
        # ------------------------------

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = np.array(tuple(map(lambda s: s is not None, batch.next_state)))

        # 現状のモデルをもとに次の状態で最大のQ値を求める
        next_state_values = np.zeros(batch_size)
        next_state_values[non_final_mask] = self.model.predict(non_final_next_states).max(1)

        # 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values.reshape(batch_size, 1) 

        # ------------------------------
        # 4. パラメータの更新
        # ------------------------------

        self.model.fit(state_batch, expected_state_action_values, epochs=1, verbose=0, batch_size=batch_size)
        
    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # eps-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            state = state.reshape(1, self.num_states)
            action = self.model.predict(state).argmax(1).reshape(1,1)
        else:
            # 0,1の行動をランダムに返す
            action = np.array([[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す

        return action

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)


class Agent:
    def __init__(self, num_states, num_actions, memory_cap):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions, memory_cap)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self, batch_size):
        '''Q関数を更新する'''
        self.brain.replay(batch_size)

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)
