#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tf
import math
import json
import random
import rospy
import subprocess
import numpy as np

from geometry_msgs.msg import Twist, Vector3, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from RL import DQN

# parameters setting
WIDTH = 1.2
FPS = 4
NUM_STEP = 3 * 60 * FPS

VEL = 0.5
OMEGA = 15.0 * math.pi / 180.0

NUM_STATES = 24
BATCH_SIZE = 32

# action list, [velocity, angle]
action_list = {
    0: [-VEL, -OMEGA],
    1: [-VEL, 0.0],
    2: [-VEL, OMEGA],
    3: [0.0, -OMEGA],
    4: [0.0, 0.0],
    5: [0.0, OMEGA],
    6: [VEL, -OMEGA],
    7: [VEL, 0.0],
    8: [VEL, OMEGA],
}

# score names
body_score_name = ('RE_B', 'RE_L', 'RE_R', 'BL_B', 'BL_L', 'BL_R')
field_score_name = ('hoge1_N', 'hoge1_S', 'hoge2_N', 'hoge2_S', 'hoge3_N', 'hoge3_S',
                    'hoge4_N', 'hoge4_S', 'hoge5_N', 'hoge5_E', 'hoge5_W', 'hoge5_S')


def get_rotation_matrix(rad):  # 座標回転行列を返す
    '''
    :param rad <float>: 回転角(rad)
    '''
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


def rotate_pose(x, y):
    '''
    :param x <float>: 位置のx成分
    :param y <float>: 位置のy成分
    '''
    pos = np.array([x, y])                                      # 現在地点
    rot = get_rotation_matrix(-45 * math.pi / 180)              # 45度回転行列の定義
    return np.dot(rot, pos)


def normalize_angle(angle):
    '''
    :param angle <float>: 角度
    '''
    angle -= 45 * math.pi / 180
    while angle > 0:
        angle -= 2 * math.pi
    while angle < 0:
        angle += 2 * math.pi

    return angle

# gazebo座標からamcl_pose座標に変換する
def convert_coord_from_gazebo_to_amcl(my_color, gazebo_x, gazebo_y):
    if my_color == 'r':
        amcl_x    =  gazebo_y
        amcl_y    = -gazebo_x
    else:
        amcl_x    = -gazebo_y
        amcl_y    =  gazebo_x
    return amcl_x, amcl_y


class DQNBot():

    def __init__(self, bot_name="DQN", color='r'):
        
        self.episode = 0
        self.name = bot_name                            # bot name
        self.step = 0                                   # time step
        #  0:自分位置_x,  1:自分位置_y,  2:自分角度_x,  3:自分角度_y,  4:自分角度_z,  5:自分角度_w
        #  6:相手位置_x,  7:相手位置_y,  8:相手角度_x,  9:相手角度_y, 10:相手角度_z, 11:相手角度_w
        self.pos = np.zeros(12)
        self.my_color = color                           # 自分の色
        self.en_color = 'b' if color == 'r' else 'r'    # 敵の色
        self.field_socre = {k: 0.0 for k in field_score_name}
        self.body_score = {k: 0.0 for k in body_score_name}
        self.my_score = 0.0
        self.en_score = 0.0
        self.d_my_score = 0.0
        self.d_en_score = 0.0

        # velocity publisher
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        # subscriber
        self.lidar_sub = rospy.Subscriber("scan", LaserScan, self.lidarCallback)
        self.war_state = rospy.Subscriber("war_state", String, self.callback_war_state, queue_size=10)

        self.training = True
        self.debug_use_gazebo_my_pos = True
        self.debug_use_gazebo_enemy_pos = True
        if self.debug_use_gazebo_my_pos is False:
            rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
        if self.debug_use_gazebo_enemy_pos is False:
            self.pos[6] = 1.0 if self.my_color == 'r' else -1.0
            self.pos[7] = 0.0
        if (self.debug_use_gazebo_my_pos is True) or (self.debug_use_gazebo_enemy_pos is True):
            rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state, queue_size=10)

        self.speed = VEL

        self.twist = Twist()
        self.twist.linear.x = self.speed
        self.twist.linear.y = 0.
        self.twist.linear.z = 0.
        self.twist.angular.x = 0.
        self.twist.angular.y = 0.
        self.twist.angular.z = 0.
        
    def getState(self):

        my_pos_x, my_pos_y = rotate_pose(self.pos[0], self.pos[1])
        my_angle = tf.transformations.euler_from_quaternion((self.pos[2], self.pos[3], self.pos[4], self.pos[5]))
        my_angle = normalize_angle(my_angle.z)

        en_pos_x, en_pos_y = rotate_pose(self.pos[6], self.pos[7])
        en_angle = tf.transformations.euler_from_quaternion((self.pos[8], self.pos[9], self.pos[10], self.pos[11]))
        en_angle = normalize_angle(en_angle.z)

        field_score = [v for v in self.field_socre.values()]
        if self.my_color == 'r':
            my_body = [v for v in self.body_score.values()[:3]]
            en_body = [v for v in self.body_score.values()[3:]]
        else:
            my_body = [v for v in self.body_score.values()[3:]]
            en_body = [v for v in self.body_score.values()[:3]]

        state = [my_pos_x, my_pos_y, my_angle, en_pos_x, en_pos_y, en_angle] + field_score + my_body + en_body

        return np.array(state).reshape(1, NUM_STATES)

    def callback_amcl_pose(self, data):
        pos = data.pose.pose.position
        ori = data.pose.pose.orientation
        self.pos[0] = pos.x
        self.pos[1] = pos.y
        self.pos[2] = ori.x
        self.pos[3] = ori.y
        self.pos[4] = ori.z
        self.pos[5] = ori.w

    def callback_model_state(self, data):
        #print('*********', len(data.pose))
        if 'red_bot' in data.name:
            index_r = data.name.index('red_bot')
        else:
            print('callback_model_state: red_bot not found')
            return
        if 'blue_bot' in data.name:
            index_b = data.name.index('blue_bot')
        else:
            print('callback_model_state: blue_bot not found')
            return
        #print('callback_model_state: index_r=', index_r, 'index_b=', index_b)
        my    = index_r if self.my_color == 'r' else index_b
        enemy = index_b if self.my_color == 'r' else index_r
        gazebo_my_x,    gazebo_my_y    = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[my   ].position.x, data.pose[my   ].position.y)
        gazebo_enemy_x, gazebo_enemy_y = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[enemy].position.x, data.pose[enemy].position.y)
        if self.debug_use_gazebo_my_pos is True:
            self.pos[0] = gazebo_my_x
            self.pos[1] = gazebo_my_y
            ori = data.pose[my].orientation
            self.pos[2] = ori.x
            self.pos[3] = ori.y
            self.pos[4]  = ori.z
            self.pos[5]  = ori.w
        if self.debug_use_gazebo_enemy_pos is True:
            self.pos[6] = gazebo_enemy_x
            self.pos[7] = gazebo_enemy_y
            ori = data.pose[enemy].orientation
            self.pos[8] = ori.x
            self.pos[9] = ori.y
            self.pos[10] = ori.z
            self.pos[11] = ori.w

    def lidarCallback(self, data):
        is_near_wall = self.isNearWall(data.ranges)
        if is_near_wall:
            self.twist.linear.x = -self.speed
        else:
            self.twist.linear.x = self.speed

    def callback_war_state(self, data):
        my_score_old = self.my_score
        en_score_old = self.en_score

        json_dict = json.loads(data.data)
        self.my_score = json_dict['scores'][self.my_color] # 自分のスコア
        self.en_score = json_dict['scores'][self.en_color] # 相手のスコア

        self.d_my_score = self.my_score - my_score_old
        self.d_en_score = self.en_score - en_score_old

        if json_dict['state'] == 'running':
            targets = json_dict['targets']
            for target in targets:
                name = target['name']
                if name in body_score_name:
                    self.body_score[name] = 1.0
                else:
                    if target['player'] == self.my_color:
                        self.field_score[name] = 1.0
                    else:
                        self.field_score[name] = -1.0

    def isNearWall(self, scan):
        if not len(scan) == 360:
            return False
        forword_scan = scan[:10] + scan[-10:]
        forword_scan = [x for x in forword_scan if x > 0.1]
        if min(forword_scan) < 0.2:
            return True
        return False

    def updatePoseTwist(self, action):
        vel, omega = action_list[action]
        self.twist.angular.z = omega
        self.twist.linear.x = vel

    def cal_reward(self):

        reward = 0.0
        
        dif_score = self.my_score - self.en_score
        if self.step > NUM_STEP:
            if dif_score > 0: reward = 10
            else: reward = -10
        else:
            reward = self.d_my_score - self.d_en_score
            
        return reward

    def restart(self):
        self.vel_pub.publish(Twist())  # 動きを止める
        self.step = 0
        self.field_socre = {k: 0.0 for k in field_score_name}
        self.body_score = {k: 0.0 for k in body_score_name}
        self.my_score = 0.0
        self.en_score = 0.0
        self.d_my_score = 0.0
        self.d_en_score = 0.0
        self.agent = DQN.Agent(num_states=NUM_STATES, num_actions=len(action_list), memory_cap=NUM_STEP)
        self.agent.brain.load_model('./weight.hdf5')
        subprocess.call('bash ../catkin_ws/src/burger_war/burger_war/scripts/reset_state.sh', shell=True)

    def strategy(self):
        r = rospy.Rate(FPS)

        self.agent = DQN.Agent(num_states=NUM_STATES, num_actions=len(action_list), memory_cap=NUM_STEP)

        state = self.getState()

        while not rospy.is_shutdown():
            
            action = self.agent.get_action(state=state, episode=self.episode)

            self.updatePoseTwist(action)
            self.vel_pub.publish(self.twist)

            if self.step > NUM_STEP:
                state_next = None
            else:
                state_next = self.getState()

            reward = self.cal_reward()
            reward = np.array([[reward]])

            self.agent.memorize(state, action, state_next, reward)
            self.agent.update_q_function(batch_size=BATCH_SIZE)
            state = state_next
            
            if self.step > NUM_STEP:
                self.agent.brain.save_model('./weight.hdf5')
                self.restart()
                state = self.getState()
                self.episode += 1
                r.sleep()
            else:
                self.step += 1
                r.sleep()


if __name__ == '__main__':

    rname = rosparam.get_param('randomRun/rname')
    rside = rosparam.get_param('randomRun/rname')
    if rname == 'red_bot' or rside == 'r': color = 'r'
    else                                 : color = 'b'
    print('****************', rname, rside, color)

    rospy.init_node('dqn_run')
    bot = DQNBot('DQN', color=color)
    bot.strategy()