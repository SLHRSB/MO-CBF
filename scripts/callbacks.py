from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os

class SpeedLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(SpeedLoggerCallback, self).__init__(verbose)        
        self.episode_speeds = []
        print(" -------------------------- SpeedLoggerCallback") 

    def _on_step(self) -> bool:        
        try:
            env = self.training_env.envs[0]
            current_speed = env.host_velocity  
            # print("speed from callbacks",env.host_velocity)
            self.episode_speeds.append(current_speed)
            # Check if an episode has ended
            done = self.locals.get('done')
            # done = env.terminal
            # print(done)
            if done:
                # print(" ------------------------------------------------- env.terminalType:", env.terminalType)
                if env.terminalType != 'Overlap!': #'episode' in self.locals:
                    mean_speed = np.mean(self.episode_speeds)
                    # print("speed from callbacks",mean_speed)
                    self.logger.record('train/00_mean_episode_speed', mean_speed)
                    self.logger.dump(self.n_calls)                
                self.episode_speeds = []
        except Exception as e:
            print(f"++++++++++++++ Error in SpeedLoggerCallback: {e}")
        return True



class DeviationLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(DeviationLoggerCallback, self).__init__(verbose)
        self.episode_deviations = []

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        # Get the current deviation
        current_deviation = env.distance_from_center  
        self.episode_deviations.append(current_deviation)

        # Check if an episode has ended
        done = self.locals.get('done')
        if done: #'episode' in self.locals:
            mean_deviation = np.mean(self.episode_deviations)
            
            # Log the mean deviation
            self.logger.record('train/00_mean_episode_deviation', mean_deviation)
            self.logger.dump(self.n_calls)
            
            # Reset the episode deviations
            self.episode_deviations = []

        return True
    


class CollisionLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CollisionLoggerCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        done = self.locals.get('done')
        if done:
            if env.terminalType != 'Overlap!':      
                collision_rate = env.num_collision / env.episode if env.episode!= 0 else env.num_collision
                self.logger.record('train/00_collision_rate', collision_rate)
                self.logger.dump(self.n_calls)
        
        return True
    


class EmgBrLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(EmgBrLoggerCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        done = self.locals.get('done')
        if done:
            number_of_EmgBr = env.num_emgBr
            self.logger.record('train/00_number_of_EmgBr', number_of_EmgBr)
            self.logger.dump(self.n_calls)
        
        return True
    



class CBFInterventionLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CBFInterventionLoggerCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        done = self.locals.get('done')
        if done:
            num_cbf_intervention = env.num_cbf_intervention
            self.logger.record('train/00_num_cbf_intervention', num_cbf_intervention)
            self.logger.dump(self.n_calls)
        
        return True



class RLThrottleActionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RLThrottleActionCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        RL_Throttle_Action = env.ACC_action[0]
        self.logger.record('train/01_RL_Throttle_Action', RL_Throttle_Action)
        self.logger.dump(self.n_calls)
        
        return True


class SafeThrottleActionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(SafeThrottleActionCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        Safe_Throttle_Action = env.safe_action[0]
        self.logger.record('train/01_Safe_Throttle_Action', Safe_Throttle_Action)
        self.logger.dump(self.n_calls)
        
        return True

class RLBrakeActionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RLBrakeActionCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        RL_Brake_Action = env.ACC_action[1]
        self.logger.record('train/02_RL_Brake_Action', RL_Brake_Action)
        self.logger.dump(self.n_calls)
        
        return True


class SafeBrakeActionCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(SafeBrakeActionCallback, self).__init__(verbose)
 
    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        Safe_Brake_Action = env.safe_action[1]
        self.logger.record('train/02_Safe_Brake_Action', Safe_Brake_Action)
        self.logger.dump(self.n_calls)
        
        return True


