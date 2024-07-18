import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import random
from device_classes import Intermittent, Uninterruptible

class EMSGymEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, data_file : pd.Series, intermittent_devices : list, uninterruptible_devices : list, episode_horizon : int, time_step_duration : float):

        '''
        This class is the environment of the EMS-ENV. It contains the attributes and methods that are specific to the environment.

        Args:
        data_file : The csv file containing the electricity prices
        intermittent_devices : A list containing all the intermittent devices that are going to exist in this Smart Home
        uninterruptible_devices : A list containing all the uninterruptible devices that are going to exist in this Smart Home
        episode_horizon : The length of the episode in hours
        time_step_duration : The duration of each time step in hours
        '''

        # Loading data from the csv file
        self.data = pd.read_csv(data_file)

        # Getting data from the csv file that are needed for the environment
        energy_price = self.data.loc[:,'Price (cents per kWh)']

        # Setting the number of moments that we want to keep the same price based on the time step duration that the user has given
        self.moments_to_keep_same_price = 1//time_step_duration

        # Repeating the same price for the moments that we want to keep the same price
        self.energy_price = pd.DataFrame(np.repeat(energy_price.values, self.moments_to_keep_same_price, axis=0))

        # Counting the number of intermittent and uninterruptible devices
        self.intermittent_n_devices = len(intermittent_devices)
        self.uninterruptible_n_devices = len(uninterruptible_devices)

        # Setting the power consumption of the intermittent devices
        self.intermittent_device_power_consumption = np.array([device.get_device_power_consumption() for device in intermittent_devices])

        # Setting the power consumption of the uninterruptible devices
        self.uninterruptible_device_power_consumption = np.array([device.get_device_power_consumption() for device in uninterruptible_devices])

        # Representing each moment as an hour of the day
        time = []
        
        for i in range(int(len(self.data)*(1//time_step_duration))):
            time.append(i % (24//time_step_duration))

        # Assigning the time array to a variable
        self.time = time

        # Assigning the time step duration to a variable
        self.time_step_duration = time_step_duration

        # Setting the duration of the uninterruptible devices
        self.uninterruptible_device_on_duration = np.array([device.get_device_on_duration() for device in uninterruptible_devices])

        # Changing the duration based on the time step duration
        self.uninterruptible_device_on_duration = self.uninterruptible_device_on_duration * (1//time_step_duration)

        # Create an array that will countdown the activation of the uninterruptible devices
        self.uninterruptible_device_on_countdown = np.zeros(self.uninterruptible_n_devices, dtype=float)

        # Setting the episode length : 24 for one day, 168 for one week etc. (if time_step_duration=1)
        self.max_steps = episode_horizon * 24 * (1//time_step_duration)

        # Setting the intermittent user transition probabilities
        self.intermittent_user_transition_probabilites = np.array([device.get_user_probabilities() for device in intermittent_devices])

        # Setting the uninterruptible user transition probabilities
        self.uninterruptible_user_transition_probabilites = np.array([device.get_user_probabilities() for device in uninterruptible_devices])

        # Setting the device penalties
        self.uninterruptible_override_penalty = np.array([device.get_device_override_penalty() for device in uninterruptible_devices])
        self.intermittent_standard_penalty = np.array([device.get_device_standard_penalty() for device in intermittent_devices])
        self.uninterruptible_standard_penalty = np.array([device.get_device_standard_penalty() for device in uninterruptible_devices])

        # Checking if the environment consists of only intermittent devices or not in order to set the appropriate action and observation spaces
        if (self.uninterruptible_n_devices!=0):
            # The action space includes the possible actions, meaning all the possible combinations of on/off devices that the user has registered
            self.action_space = spaces.Dict({
                'intermittent_devices' : spaces.Box(low=0, high=1, shape=(self.intermittent_n_devices,)),
                'uninterruptible_devices' : spaces.Box(low=0, high=1, shape=(self.uninterruptible_n_devices,)),
            })
        
            # The observation space includes the knowledge of which devices are on or off and the energy price of the current timestep 
            self.observation_space = spaces.Dict({
                'intermittent_devices': spaces.Box(low=0, high=1, shape=(self.intermittent_n_devices,)),
                'uninterruptible_devices': spaces.Box(low=0, high=1, shape=(self.uninterruptible_n_devices,)),
                'uninterruptible_devices_duration_countdown' : spaces.Box(low=0, high=max(self.uninterruptible_device_on_duration), shape=(self.uninterruptible_n_devices,)), # We want the agent to see how much time is left for each uninterruptible device
                'energy_price' : spaces.Box(low=-10000, high=10000, dtype=float),
                'time' : spaces.Box(low=0, high=24//time_step_duration, dtype=float)
            })

        else:
            # The action space includes the possible actions, meaning all the possible combinations of on/off devices that the user has registered
            self.action_space = spaces.Box(low=0, high=1, shape=(self.intermittent_n_devices,),dtype=float)
        
            # The observation space includes the knowledge of which devices are on or off and the energy price of the current timestep 
            self.observation_space = spaces.Dict({
                'devices': spaces.Box(low=0, high=1, shape=(self.intermittent_n_devices,)),
                'energy_price' : spaces.Box(low=-10000, high=10000,dtype=float),
                'time' : spaces.Box(low=0, high=24//time_step_duration, dtype=float)
            })

        # Calling the reset function in order to reset the environment
        self.reset()

        print("Environment successfully initialized")

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        
        if hasattr(self, 'rng'):
            self.rng.seed(seed)

        return [seed]
    
    def state_transition(self, observation, action, user_transition_probabilites):
        
        # Create arrays to represent the different cases of the transition probabilities
        omega = user_transition_probabilites[:, 0]
        p = user_transition_probabilites[:, 1]
        theta = user_transition_probabilites[:, 2]
        q = user_transition_probabilites[:, 3]

        # We round the action to 0 or 1 because we currently accept only On/Off states
        action = np.round(action)

        # We also round the observation
        observation = np.round(observation)

        # Create boolean masks for the different cases of the transition probabilities
        on_to_on_mask = (action == 1) & (observation == 1)
        on_to_off_mask = (action == 0) & (observation == 1)
        off_to_off_mask = (action == 0) & (observation == 0)
        off_to_on_mask = (action == 1) & (observation == 0)

        # Initialize the new_states array
        new_states = np.zeros(user_transition_probabilites.shape[0], dtype=int)

        new_states[on_to_on_mask] = np.random.binomial(1, omega[on_to_on_mask]) 
        new_states[on_to_off_mask] = np.random.binomial(0, p[on_to_off_mask]) 
        new_states[off_to_off_mask] = np.random.binomial(1, 1 - theta[off_to_off_mask]) 
        new_states[off_to_on_mask] = np.random.binomial(1, q[off_to_on_mask]) 
       
        return new_states
    
    def step(self, action):

        # Advancing the time by one timestep
        self.current_time += 1

        if(self.uninterruptible_n_devices!=0):

            # Saving the old state of the uninterruptible devices
            self.old_uninterruptible_state = self.observation['uninterruptible_devices']

            # Calculating the device state, i.e. we see which changes the user accepts
            self.intermittent_devices_now = self.state_transition(self.observation['intermittent_devices'], action['intermittent_devices'], self.intermittent_user_transition_probabilites)
            self.uninterruptible_devices_now = self.state_transition(self.observation['uninterruptible_devices'], action['uninterruptible_devices'], self.uninterruptible_user_transition_probabilites)
            
            self.temp_uninterruptible_countdown = self.uninterruptible_device_on_countdown.copy()

            # Updating the countdown for each uninterruptible device
            for i in range(len(self.observation['uninterruptible_devices'])):
                if self.uninterruptible_devices_now[i] == 1:
                    if self.old_uninterruptible_state[i] == 0:  # If the device was off and now is on we have to activate it
                        self.temp_uninterruptible_countdown[i] = self.uninterruptible_device_on_duration[i]
                    self.temp_uninterruptible_countdown[i] -= 1  # Decrease the countdown
                    if (self.temp_uninterruptible_countdown[i]<=0):
                        self.temp_uninterruptible_countdown[i] = 0
                        self.uninterruptible_devices_now[i] = 0

            # Getting the energy price of the current timestep
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]
            
            # Calculating the reward for uninterruptible devices
            uninterruptible_reward = np.sum(
                np.where(
                    (np.round(self.old_uninterruptible_state) == 1) & (np.round(action['uninterruptible_devices']) == 0) & (self.observation['uninterruptible_devices_duration_countdown'] > 0),
                    -np.sum(self.uninterruptible_override_penalty),
                    np.where(
                        np.round(self.uninterruptible_devices_now) == np.round(action['uninterruptible_devices']), 
                            -self.energy_price_now * self.uninterruptible_device_power_consumption * self.uninterruptible_devices_now * self.time_step_duration,
                            (-self.energy_price_now * self.uninterruptible_device_power_consumption * self.uninterruptible_devices_now * self.time_step_duration) -np.sum(self.uninterruptible_standard_penalty) 
                    )
                )
            )
            
            # Calculating the reward for intermittent devices
            intermittent_reward = np.sum(
                np.where(
                    np.round(self.intermittent_devices_now) == np.round(action['intermittent_devices']), 
                        -self.energy_price_now * self.intermittent_device_power_consumption * self.intermittent_devices_now * self.time_step_duration, 
                        (-self.energy_price_now * self.intermittent_device_power_consumption * self.intermittent_devices_now * self.time_step_duration) -np.sum(self.intermittent_standard_penalty) 
                )
            )
            
            # Calculating the total reward
            self.reward = intermittent_reward + uninterruptible_reward

            # Setting the new Time
            self.time_now = self.time[self.current_time]
            
            # Setting up the current observation
            self.observation = {
                'intermittent_devices': np.array(self.intermittent_devices_now), 
                'uninterruptible_devices': np.array(self.uninterruptible_devices_now),
                'uninterruptible_devices_duration_countdown' :  np.array(self.temp_uninterruptible_countdown),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }

            # Calculating useful data for the info dictionary
            self.intermittent_device_reward_counter += intermittent_reward
            self.uninterruptible_device_reward_counter += uninterruptible_reward

            intermittent_kwh_consumption = np.sum(np.where(np.round(self.intermittent_devices_now)==1,self.intermittent_device_power_consumption,0))
            uninterruptible_kwh_consumption = np.sum(np.where(np.round(self.uninterruptible_devices_now)==1,self.uninterruptible_device_power_consumption,0))

            current_kwh_consumption = intermittent_kwh_consumption + uninterruptible_kwh_consumption

            self.plot_intermittent_kwh.append(intermittent_kwh_consumption)
            self.plot_uninterruptible_kwh.append(uninterruptible_kwh_consumption)

        else: # If we dont have uninterruptible devices

            # Calculating the device state, i.e. we see which changes the user accepts
            self.intermittent_devices_now = self.state_transition(self.observation['devices'], action, self.intermittent_user_transition_probabilites)

            # Getting the energy price of the current timestep
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]

            # Calculating the reward
            self.reward = np.sum(
                np.where(
                    np.round(self.intermittent_devices_now) == np.round(action), 
                        -self.energy_price_now * self.intermittent_device_power_consumption * self.intermittent_devices_now * self.time_step_duration, 
                        (-self.energy_price_now * self.intermittent_device_power_consumption * self.intermittent_devices_now * self.time_step_duration) -np.sum(self.intermittent_standard_penalty) 
                )
            )

            # Setting the new Time
            self.time_now = self.time[self.current_time]

            # Setting up the current observation
            self.observation = {
            'devices': np.array(self.devices_now), 
            'energy_price' : np.array([self.energy_price_now]),
            'time' : np.array([self.time_now])
            }

            # Calculating useful data for the info dictionary
            current_kwh_consumption = np.sum(np.where(np.round(self.devices_now)==1,self.intermittent_device_power_consumption*self.time_step_duration,0))


        # Appending the data to the plot arrays
        self.plot_price.append(self.energy_price_now)
        self.plot_kwh.append(current_kwh_consumption)
        self.plot_time.append(self.current_time)

        # Checking if we have reached the end of the episode horizon
        # If we have reached the end of the episode horizon
        if self.current_time >= self.max_steps:

            self.done = True

            if(self.uninterruptible_n_devices!=0):
            # We set the done flag as True to indicate that that the episode ended
                info = {"intermittent_device_reward_total" : self.intermittent_device_reward_counter,
                    "uninterruptible_device_reward_total" : self.uninterruptible_device_reward_counter,
                    "kwh_device_history" : self.plot_kwh,
                    "kwh_intermittent_device_history" : self.plot_intermittent_kwh,
                    "kwh_uninterruptible_device_history" : self.plot_uninterruptible_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
            else:
                info = {"kwh_device_history" : self.plot_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
        else:

            if(self.uninterruptible_n_devices!=0):
                info = {"intermittent_device_reward_total" : self.intermittent_device_reward_counter,
                    "uninterruptible_device_reward_total" : self.uninterruptible_device_reward_counter,
                    "kwh_device_history" : self.plot_kwh,
                    "kwh_intermittent_device_history" : self.plot_intermittent_kwh,
                    "kwh_uninterruptible_device_history" : self.plot_uninterruptible_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
            else:
                info = {"kwh_device_history" : self.plot_kwh,
                    "price_history" : self.plot_price,
                    "time" : self.plot_time}
        
        truncated = False

        return self.observation, float(self.reward), self.done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        
        if (self.uninterruptible_n_devices!=0):
            # Setting time at 0
            self.current_time = 0

            # Setting reward at 0
            self.reward = 0

            # Setting done as False in order to declare that the environment has not reached the end
            self.done = False

            # Initializing the first state with all devices off
            self.intermittent_devices_now = np.zeros(self.intermittent_n_devices)
            self.uninterruptible_devices_now = np.zeros(self.uninterruptible_n_devices)

            # Setting the first energy price
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]

            # Setting the current time
            self.time_now = self.time[self.current_time]
            
            # Setting up counters for info dictionary variables
            self.intermittent_device_reward_counter = 0
            self.uninterruptible_device_reward_counter = 0

            # Setting up the current observation
            self.observation = {
                'intermittent_devices': np.array(self.intermittent_devices_now),
                'uninterruptible_devices': np.array(self.uninterruptible_devices_now),
                'uninterruptible_devices_duration_countdown' :  np.array(self.uninterruptible_device_on_countdown),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }
        
        else:

            # Setting time at 0
            self.current_time = 0

            # Setting reward at 0
            self.reward = 0

            # Setting done as False in order to declare that the environment has not reached the end
            self.done = False

            # Setting the current time
            self.time_now = self.time[self.current_time]

            # Initializing the first state with all devices off
            self.devices_now = np.zeros(self.intermittent_n_devices)

            # Setting the first energy price
            self.energy_price_now = self.energy_price.iloc[self.current_time][0]
            
            # Setting up the current observation
            self.observation = {
                'devices': np.array(self.devices_now),
                'energy_price' : np.array([self.energy_price_now]),
                'time' : np.array([self.time_now])
            }

        # Initializing the plot arrays for the price, kwh and time
        self.plot_price = []
        self.plot_kwh = []
        self.plot_intermittent_kwh = []
        self.plot_uninterruptible_kwh = []
        self.plot_time = []
        
        # Checking if there are uninterruptible devices in order to set the appropriate info dictionary  
        if(self.uninterruptible_n_devices!=0):
            info = {"intermittent_device_reward_total" : self.intermittent_device_reward_counter,
                "uninterruptible_device_reward_total" : self.uninterruptible_device_reward_counter,
                "kwh_device_history" : self.plot_kwh,
                "price_history" : self.plot_price,
                "time" : self.plot_time}
        else:
            info = {"kwh_device_history" : self.plot_kwh,
                "price_history" : self.plot_price,
                "time" : self.plot_time}

        return self.observation, info

def main():

    # Assigning data file
    data_file = "prices_for_one_day_inference.csv"

    # Creating the intermittent devices
    intermittent_user_probabilities = np.array([0.90, 0.90, 0.90, 0.90])
    
    intermittent_device_penalty = 100
    
    intermittent_device_1 = Intermittent(name = "intermittent_device_1", device_power_consumption = 1, 
                                         user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    intermittent_device_2 = Intermittent(name = "intermittent_device_2", device_power_consumption = 2.5, 
                                        user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    intermittent_device_3 = Intermittent(name = "intermittent_device_3", device_power_consumption = 0.07,
                                        user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    intermittent_device_4 = Intermittent(name = "intermittent_device_4", device_power_consumption = 0.07,
                                        user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    intermitent_device_5 = Intermittent(name = "intermittent_device_5", device_power_consumption = 3,
                                        user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    intermittent_device_6 = Intermittent(name = "intermittent_device_6", device_power_consumption = 0.07,
                                        user_probabilities = intermittent_user_probabilities, device_standard_penalty = intermittent_device_penalty)
    
    array_of_intermittent_devices = [intermittent_device_1, intermittent_device_2, intermittent_device_3, intermittent_device_4, intermitent_device_5, intermittent_device_6]

    # Creating the uninterruptible devices

    uninterruptible_user_probabilities = np.array([0.90, 0.90, 0.90, 0.90])

    uninterruptible_device_standard_penalty = 10

    uninterruptible_device_override_penalty = 100

    uninterruptible_device_1 = Uninterruptible(name = "uninterruptible_device_1", device_power_consumption = 1.3,
                                                user_probabilities = uninterruptible_user_probabilities, device_standard_penalty = uninterruptible_device_standard_penalty, 
                                                device_on_duration = 2.5, device_override_penalty = uninterruptible_device_override_penalty)
    
    uninterruptible_device_2 = Uninterruptible(name = "uninterruptible_device_2", device_power_consumption = 0.5,
                                                user_probabilities = uninterruptible_user_probabilities, device_standard_penalty = uninterruptible_device_standard_penalty, 
                                                device_on_duration = 1, device_override_penalty = uninterruptible_device_override_penalty)
    
    uninterruptible_device_3 = Uninterruptible(name = "uninterruptible_device_3", device_power_consumption = 2.4,
                                                user_probabilities = uninterruptible_user_probabilities, device_standard_penalty = uninterruptible_device_standard_penalty, 
                                                device_on_duration = 0.5, device_override_penalty = uninterruptible_device_override_penalty)
    
    array_of_uninterruptible_devices = [uninterruptible_device_1, uninterruptible_device_2, uninterruptible_device_3]

    # Creating the environment
    accepting_train_env = EMSGymEnv(data_file = data_file, intermittent_devices = array_of_intermittent_devices, 
                    uninterruptible_devices = array_of_uninterruptible_devices, episode_horizon = 1, time_step_duration = 0.5)

    for i in range(8):
        accepting_train_env.step({'uninterruptible_devices' : [0.51,0.611,0.4], 'intermittent_devices' : [0.8527126,0.8528148, 0.3818789, 0.4139895, 0.4671028,0.3]})


if __name__ == "__main__":
    main()