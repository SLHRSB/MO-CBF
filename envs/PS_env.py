import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from .PS_env_helper import Reciver_UDP2, Transmitter_UDP_2
from .mitm import MitM
from .SafetyLayer import DoubleCBF


class Prescan_env(gym.Env):
    """
    PreScan ACC environment for PPO training/testing.

    The environment receives vehicle/sensor data through UDP,
    applies the RL action, optionally passes it through a safety module,
    and sends the final throttle/brake command back to PreScan.
    """

    def __init__(
        self,
        train=False,
        with_vissim=False,
        short_run=True,
        safety_module=2,   # 0: none, 1: rule-based, 2: CBF
        learning_emgbr=False,
        attack=False,
        attack_type="DOS",
        dos_duration=3,
        dos_start_step=7,
        noise_level=0.5,
        print_log=False,
    ):
        super().__init__()

        self.train = train
        self.with_vissim = with_vissim
        self.short_run = short_run
        self.safety_module = safety_module
        self.learning_EmgBr = learning_emgbr
        self.attack = attack
        self.print_log = int(print_log)

        self.episode = 0
        self.iter_in_episod = 0
        self.num_lane_chane = 0

        self.reset_flag = 0
        self.terminal = 0
        self.terminalType = None
        self.collision_detected = 0
        self.long = 0
        self.end_road = 0

        self.num_collision = 0
        self.num_long = 0
        self.num_overlap = 0
        self.num_emgBr = 0
        self.emgBr_flag = 0
        self.cbf_flag = 0
        self.num_cbf_intervention = 0

        self.speed = 0
        self.current_desired_velocity = 10

        self.mitm = MitM(
            attack_type=attack_type,
            noise_level=noise_level,
            dos_duration=dos_duration,
            dos_start_step=dos_start_step,
        )

        # UDP communication with PreScan
        self.env_info_udp = Reciver_UDP2("env_info_udp", 8031)
        self.env_info_udp.build()
        self.send_action_UDP = Transmitter_UDP_2("send_action", 8032)

        # One-dimensional ACC action:
        # action >= 0 -> throttle
        # action < 0  -> brake
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32,
        )

        # Observation:
        # [ego speed, leader distance, leader speed, episode progress]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.action = 0
        self.ACC_action = np.array([0.0, 0.0], dtype=np.float32)
        self.safe_action = np.array([0.0, 0.0], dtype=np.float32)

        self.state = np.array(self.observation(), dtype=np.float32)

        print(
            "-------------------------------- "
            "PreScan ACC Environment created "
            "--------------------------------"
        )

    def close(self):
        try:
            self.env_info_udp.close()
            self.send_action_UDP.close()
            print("Environment closed.")
        except Exception as e:
            print(f"Error when closing environment: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.iter_in_episod = 0
        self.num_lane_chane = 0
        self.current_desired_velocity = 10

        self.terminal = 0
        self.terminalType = None
        self.collision_detected = 0
        self.long = 0
        self.end_road = 0

        self.emgBr_flag = 0
        self.cbf_flag = 0

        # Reset PreScan scenario
        self.reset_flag = 1
        self.send_action_UDP.send_data(
            0,
            self.current_desired_velocity,
            0,
            0,
            self.reset_flag,
        )

        self.reset_flag = 0
        self.send_action_UDP.send_data(
            0,
            self.current_desired_velocity,
            0,
            0,
            self.reset_flag,
        )

        self.state = self.observation()
        self.action = 0
        self.ACC_action = np.array([0.0, 0.0], dtype=np.float32)
        self.safe_action = np.array([0.0, 0.0], dtype=np.float32)

        # If collision happens immediately after reset, reset once more
        if self.collision_detected:
            self.reset_flag = 1
            self.send_action_UDP.send_data(
                0,
                self.current_desired_velocity,
                0,
                0,
                self.reset_flag,
            )

            time.sleep(0.1)

            self.reset_flag = 0
            self.send_action_UDP.send_data(
                0,
                self.current_desired_velocity,
                0,
                0,
                self.reset_flag,
            )

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        action_value = float(np.asarray(action).item())
        self.action = action_value

        if action_value >= 0:
            self.ACC_action[0] = action_value
            self.ACC_action[1] = 0.0
        else:
            self.ACC_action[0] = 0.0
            self.ACC_action[1] = -action_value

        self.iter_in_episod += 1
        reward_step = 0.0

        self.emgBr_flag = 0
        self.cbf_flag = 0

        if self.safety_module == 1:
            self.safe_action = np.array(self.Rule_based_safety(), dtype=np.float32)

        elif self.safety_module == 2:
            self.safety = DoubleCBF(
                self.ACC_action[0],
                self.ACC_action[1],
                self.leader_dis,
                self.host_velocity,
                self.leader_v,
                self.host_engine_RPM,
            )

            self.safe_action = np.array(
                self.safety.control_barrier_function(),
                dtype=np.float32,
            )

            if not np.allclose(self.ACC_action, self.safe_action):
                self.cbf_flag = 1
                self.num_cbf_intervention += 1

        else:
            self.safe_action = self.ACC_action.copy()

        self.takeAction()

        self.terminal, self.terminalType = self.terminalCheck()

        reward_step += self.reward()

        if not self.terminal:
            self.state = self.observation()

        return np.array(self.state, dtype=np.float32), reward_step, bool(self.terminal), False, {}

    def observation(self):
        data = self.env_info_udp.get()
        host_position = data["Pos"]

        self.x_host = float(host_position["x"].strip())

        self.host_velocity = float(data["Vel"].strip())
        self.host_acceleration = float(data["Acc"].strip())
        self.host_jerk = float(data["Jerk"].strip())
        self.host_engine_torque = float(data["E_torq"].strip())
        self.host_engine_RPM = float(data["RPM"].strip())
        self.host_engine_power = float(data["E_power"].strip())

        self.leader_dis = float(data["R001"].strip())
        if self.leader_dis == 0:
            self.leader_dis = 150.0

        self.leader_v = float(data["V001"].strip())

        if int(data["if_c"].strip()) == 1:
            collision_id_1 = data["C_1"].strip()
            collision_id_2 = data["C_2"].strip()

            if (
                collision_id_1 == "11"
                or collision_id_2 == "11"
                or (collision_id_1 == "0" and collision_id_2 == "0")
            ):
                self.collision_detected = 1
            else:
                self.collision_detected = 0
        else:
            self.collision_detected = 0

        state = np.array(
            [
                self.host_velocity / 30.0,
                self.leader_dis / 150.0,
                self.leader_v / 30.0,
                self.iter_in_episod / 100.0,
            ],
            dtype=np.float32,
        )

        if self.attack:
            state = self.mitm.modify_observation(state, self.iter_in_episod)

        state = np.clip(state, 0.0, 1.0)

        return state

    def reward(self):
        reward = -0.05

        max_reward_speed = 1.0

        if self.host_velocity <= 5:
            reward_speed = -max_reward_speed * self.host_velocity / 10.0
        elif self.host_velocity <= 25:
            reward_speed = max_reward_speed * self.host_velocity / 10.0
        else:
            reward_speed = -max_reward_speed * self.host_velocity / 10.0

        reward += reward_speed

        if self.terminal:
            if self.terminalType == "End of the road!":
                reward += 5.0

            elif self.terminalType == "Collision!":
                reward -= 10.0
                print("Collision!")

            elif self.terminalType == "Took Too Long!":
                reward -= 10.0

            elif self.terminalType == "Overlap!":
                reward += 0.0

        if self.train and self.safety_module == 1 and self.emgBr_flag:
            reward -= (
                (2 * self.host_velocity - self.leader_dis)
                / (2 * self.host_velocity)
                * 1.5
            )

        if self.train and self.safety_module == 2 and self.cbf_flag:
            penalty = 10.0 * np.linalg.norm(
                self.ACC_action - self.safe_action,
                ord=1,
            )
            reward -= penalty

        return float(reward)

    def terminalCheck(self):
        self.terminal = 0
        self.terminalType = None

        if self.short_run:
            goal_dis = 135
            max_iter_in_episode = 250
        else:
            goal_dis = 1000
            max_iter_in_episode = 1000

        overlap_dis = 45 if self.with_vissim else 5

        if int(self.x_host) > goal_dis:
            self.terminal = 1
            self.terminalType = "End of the road!"
            self.end_road = 1
            self.reset_flag = 1
            self.episode += 1

            if self.print_log:
                print("Success! terminal =", self.terminal)

        if int(self.collision_detected):
            if int(self.x_host) > overlap_dis:
                self.terminal = 1
                self.terminalType = "Collision!"
                self.num_collision += 1
                self.reset_flag = 1
                self.episode += 1

                print("Number of collisions so far:", self.num_collision)

                if self.print_log:
                    print("Collision! terminal =", self.terminal)

            else:
                self.terminal = 1
                self.terminalType = "Overlap!"
                self.num_overlap += 1
                self.reset_flag = 1

                if self.print_log:
                    print("Overlap! terminal =", self.terminal)

        if self.iter_in_episod > max_iter_in_episode:
            self.terminal = 1
            self.terminalType = "Took Too Long!"
            self.long = 1
            self.num_long += 1
            self.reset_flag = 1
            self.episode += 1

            if self.print_log:
                print("Too Long! terminal =", self.terminal)

        return self.terminal, self.terminalType

    def Rule_based_safety(self):
        if self.leader_dis <= 2 * self.host_velocity:
            self.safe_action = np.array([0.0, 1.0], dtype=np.float32)
            self.num_emgBr += 1
            print("Emergency braking!")

            if self.learning_EmgBr:
                self.emgBr_flag = 1
        else:
            self.safe_action = self.ACC_action.copy()

        return self.safe_action

    def takeAction(self):
        off_set = 0
        desired_velocity = self.host_velocity

        if self.safety_module in [1, 2]:
            throttle_flag = float(self.safe_action[0])
            brake_flag = float(self.safe_action[1])
        else:
            throttle_flag = float(self.ACC_action[0])
            brake_flag = float(self.ACC_action[1])

        self.send_action_UDP.send_data(
            off_set,
            desired_velocity,
            throttle_flag,
            brake_flag,
            self.reset_flag,
        )
