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

from RLmodule.Qlearning import Agent, Brain

# parameters setting
WIDTH = 1.2
FPS = 4
NUM_STEP = 3 * 60 * FPS

NUM_DISCRIZED = 24
NUM_STATES = NUM_DISCRIZED**3

VEL = 0.5
OMEGA = 15.0 * math.pi / 180.0

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

state_range = [
    [-WIDTH, WIDTH],
    [-WIDTH, WIDTH],
    [0, 2 * math.pi]
]


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
    angle -= -45 * math.pi / 180
    while angle > 0:
        angle -= 2 * math.pi
    while angle < 0:
        angle += 2 * math.pi

    return angle


class QlearningBot():
    '''
    戦略：自分の姿勢を状態とし、得点したときにプラスの報酬を与える。
    必要な情報：自己位置、得点
    '''

    def __init__(self, bot_name="QL"):
        # bot name
        self.name = bot_name
        self.step = 0

        self.pos = np.zeros(6)
        self.score = 0
        self.d_score = 0

        # velocity publisher
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        # subscriber
        self.pose_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.poseCallback)
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.lidarCallback)

        self.speed = VEL

        self.twist = Twist()
        self.twist.linear.x = self.speed
        self.twist.linear.y = 0.
        self.twist.linear.z = 0.
        self.twist.angular.x = 0.
        self.twist.angular.y = 0.
        self.twist.angular.z = 0.

        self.state = self.getState()
        self.reward = 0

    def poseCallback(self, data):
        pos = data.pose.pose.position
        ori = data.pose.pose.orientation
        self.pos[0] = pos.x
        self.pos[1] = pos.y
        self.pos[2] = ori.x
        self.pos[3] = ori.y
        self.pos[4] = ori.z
        self.pos[5] = ori.w

    def lidarCallback(self, data):
        is_near_wall = self.isNearWall(data.ranges)
        if is_near_wall:
            self.twist.linear.x = -self.speed
        else:
            self.twist.linear.x = self.speed

    def callback_war_state(self, data):
        score_old = self.score
        json_dict = json.loads(data.data)
        self.score = json_dict['scores']['r']  # 自分のスコア
        self.d_score = self.score - score_old

    def isNearWall(self, scan):
        if not len(scan) == 360:
            return False
        forword_scan = scan[:10] + scan[-10:]
        forword_scan = [x for x in forword_scan if x > 0.1]
        if min(forword_scan) < 0.2:
            return True
        return False

    def getState(self):

        pos_x, pos_y = rotate_pose(self.pos[0], self.pos[1])
        angle = tf.transformations.euler_from_quaternion((self.pos[2], self.pos[3], self.pos[4], self.pos[5]))
        angle = normalize_angle(angle.z)

        state = [pos_x, pos_y, angle]

        rospy.Subscriber("war_state", String, self.callback_war_state, queue_size=10)

        return state

    def updatePoseTwist(self, action):
        vel, omega = action_list[action]
        self.twist.angular.z = omega
        self.twist.linear.x = vel

    def restart(self):
        self.vel_pub.publish(Twist())  # 動きを止める
        self.step = 0
        self.score = 0
        self.d_score = 0
        self.reward = 0
        brain = Brain(num_states=NUM_STATES, num_actions=len(action_list),
                      state_bins=state_range, num_digitized=NUM_DISCRIZED)
        self.agent = Agent(brain=brain)
        self.agent.load_q_table('./q_table.csv')
        subprocess.call('bash ../catkin_ws/src/burger_war/burger_war/scripts/reset_state.sh', shell=True)

    def strategy(self):
        r = rospy.Rate(FPS)

        brain = Brain(num_states=NUM_STATES, num_actions=len(action_list),
                      state_bins=state_range, num_digitized=NUM_DISCRIZED)
        self.agent = Agent(brain=brain)

        while not rospy.is_shutdown():
            self.step += 1

            next_state = self.getState()
            self.reward += self.d_score
            self.agent.update_Q_function(self.state, action, self.reward, next_state)
            self.state = next_state
            action = self.agent.get_action(observation=self.state, episode=self.step)
            self.updatePoseTwist(action)

            self.vel_pub.publish(self.twist)

            if self.step > NUM_STEP:
                self.agent.save_q_table('./q_table.csv')
                self.restart()
                r.sleep()

            r.sleep()


if __name__ == '__main__':
    rospy.init_node('q_run')
    bot = QlearningBot('QL')
    bot.strategy()
