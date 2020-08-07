import numpy as np


class Agent:
    '''
    エージェントクラス

    usage
    > agent = Agent()
    > agent.update_Q_function()
    > action = agent.get_action()

    :var brain <class>: Brainクラス
    '''

    def __init__(self, brain):
        '''
        :param brain <class>: Brainクラス
        '''
        self.brain = brain

    def update_Q_function(self, observation, action, reward, observation_next):
        '''
        Q関数の更新
        :param observation <ndarray> or <list>: 観測した状態のindex
        :param action <int>: 行動のindex
        :param reward <int> or <float>: 報酬
        :param observation_next <ndarray> or <list>: 次の観測（した状態のindex）
        '''
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, episode):
        '''
        行動の決定
        :param observation <ndarray> or <list>: 観測した状態のindex
        :param episode <int>: エピソード数（ステップ数）
        '''
        action = self.brain.decide_action(observation, episode)
        return action

    def save_q_table(self, path):
        np.savetxt(path, self.brain.q_table, fmt='%.8f')

    def load_q_table(self, path):
        q = np.loadtxt(path)
        self.brain.q_table = q


class Digitize:
    '''
    離散化クラス

    usage
    > digitizer = Digitize()
    > digitizer.digitize_state()

    :var state_bins <ndarray> or <list>: 状態の各要素の最小値、最大値のリスト（２次元配列）
    :var num_digitized <int>: 各状態の離散値への分割数
    '''

    def __init__(self, state_bins, num_digitized):
        '''
        :param state_bins <ndarray> or <list>: 状態の各要素の最小値、最大値のリスト（２次元配列）
        :param num_digitized <int>: 各状態の離散値への分割数
        '''
        self.state_bins = state_bins
        self.num_digitized = num_digitized

    def bins(self, clip_min, clip_max, num):
        '''
        離散値のindexを生成
        :param clip_min <float>: 離散値の最小値
        :param clip_max <float>: 離散値の最大値
        :param num <int>: 離散値の分割数
        '''
        return np.linspace(clip_min, clip_max, num+1)[1:-1]

    def digitize_state(self, observation):
        '''
        各状態を離散化
        :param observation <ndarray> or <list>: 観測した状態の値
        '''
        digitized = []
        for obs, sb in zip(observation, self.state_bins):
            obs_dig = np.digitize(obs, bins=self.bins(sb[0], sb[1], self.num_digitized))
            digitized.append(obs_dig)
        return sum([x * (self.num_digitized**i) for i, x in enumerate(digitized)])


class Brain:
    '''
    エージェントが持つ脳となるクラス、Q学習を実行する

    usage
    > brain = Brain()
    > brain.update_Q_table()
    > action = brain.decide_action()

    :var num_actions <int>: 行動の数
    :var alpha <float>: Q学習の学習係数
    :var gamma <float>: 時間割引率、0 < gamma < 1
    :var digitizer <class>: 状態を離散化するクラス
    :var q_table <ndarray>(2D): Q値のテーブル、q_table.shape = (num_states, num_actions)
    '''

    def __init__(self, num_states, num_actions, state_bins, num_digitized, alpha=0.5, gamma=0.99):
        '''
        :param num_states <int>: 状態の数
        :param num_actions <int>: 行動の数
        :param state_bins <ndarray> or <list>: 状態の各要素の最小値、最大値のリスト（２次元配列）
        :param num_digitized <int>: 離散化の分割数
        :param alpha <float>: Q学習の学習係数
        :param gamma <float>: 時間割引率、0 < gamma < 1
        '''
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.digitizer = Digitize(state_bins, num_digitized)
        self.q_table = np.random.uniform(low=0, high=1, size=(num_digitized**num_states, num_actions))

    def update_Q_table(self, observation, action, reward, observation_next):
        '''
        QテーブルをQ学習により更新
        :param observation <ndarray> or <list>: 観測した状態のindex
        :param action <int>: 行動のindex
        :param reward <int> or <float>: 報酬
        :param observation_next <ndarray> or <list>: 次の観測（した状態のindex）
        '''
        state = self.digitizer.digitize_state(observation)
        state_next = self.digitizer.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = (1.0 - self.alpha) * self.q_table[state, action] + \
            alpha * (reward + self.gamma * Max_Q_next)

    def decide_action(self, observation, episode):
        '''
        eps-greedy法で徐々に最適行動のみを採用する
        :param observation <ndarray> or <list>: 観測した状態のindex
        :param episode <int>: エピソード数（ステップ数）
        '''
        state = self.digitizer.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)

        return action
