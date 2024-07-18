# EMS-env: A Reinforcement Learning framework for residential energy efficiency recommendations

## Description

This repository serves as a comprehensive guide and resource hub for understanding, experimenting with, and implementing RL models in the context of household energy optimization.

_EMS-env_ is a custom Gymnasium environment for residential energy efficiency recommendations.

_EMS-env paper: S. Chadoulos, O. Diamantopoulos, I. Koutsopoulos, G.C. Polyzos, and N. Ipiotis. "EMS-env: A Reinforcement Learning Framework for Residential Energy Efficiency Recommendations" (To be presented at the 2024 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids, Oslo Nowray)._

## Contents

- **/data_csv**: Contains all the data collected through experiments, which is used in the **wandb_database_plots.ipynb** notebook
- **/plots**: Contains all the plots that are produced by running the **wandb_database_plots.ipynb** notebook
- **/prudent_environment_logs**: Contains the logs of a trained agent on the Prudent environment version that can be used to execute the **rl_agent_testing_notebook.ipynb** notebook
- **device_classes.py**: Contains the implementation of the Intermittent and Uninterruptible device classes
- **EMS_Gym_Env**: Contains the custom Gymnasium environment
- **LICENCE.txt**: The licence file
- **nyiso_hourly_prices.csv**: Contains a year of historical price data from https://www.nyiso.com/energy-market-operational-data
- **nyiso_hourly_prices_for_inference.csv**: Contains a week of historical price data which has not been used in the training process and is used for inference
- **requirements.txt**: Contains the required libraries with their specific versions needed to run the repository files
- **rl_agent_testing_notebook.ipynb**: A notebook that loads the trained agents and tests them on data that they have not encountered during training
- **rl_agent_training_notebook.ipynb**: A notebook that carries out the RL agent training in different environment setups
- **wandb_database_plots.ipynb** : A notebook that uses experimental data gathered from the training process and constructs the respective plots, some of which were presented in the paper


## Setup instructions

1. [Download Anaconda](https://www.anaconda.com/download)
2. Open the Anaconda Prompt
3. Execute: `conda create -n your_env_name python=3.10.12` to create a virtual Conda environment with the specified Python version
4. Activate the virtual Conda environment: `conda activate your_env_name`
5. Go to the github repo directory `cd path/to/directory`
6. Install required packages: `pip install -r requirements.txt`
7. Follow the Weights and Biases sign up and login process: https://docs.wandb.ai/quickstart  
8. Create a new project in Weights and Biases named env_paper_experiments
9. Run the **rl_agent_training_notebook.ipynb** notebook to train the agents

## EMSGymEnv class initialization instructions
- **data_file**: Path to the csv file containing the electricity prices (it can be replaced with any other csv file containing energy prices)
- **intermittent_devices**: An array that contains Intermittent class objects
- **uninterruptible_devices**: An array that contains Uninterruptible class objects
- **episode_horizon**: Hours of learning per episode for the agent
- **time_step_duration**: Duration of each time step in hours (e.g 0.5 for 30 minutes)

## Device initialization instructions

### Intermittent devices
- **name**: The name of the device
- **device_power_consumption**: The power consumption of the device in kiloWatts
- **user_probabilities**: The ω, p, θ, q probabilities for the intermittent device
- **device_standard_penalty**: The penalty issued to the agent each time it proposes an action for this specific device that the user does not approve

### Uninterruptible devices
- **name**: The name of the device
- **device_power_consumption**: The power consumption of the device in kiloWatts
- **user_probabilities**: The ω, p, θ, q probabilities for the intermittent device
- **device_standard_penalty**: The penalty issued to the agent each time it proposes an action for this specific device that the user does not approve
- **device_on_duration**: The duration the device has to stays on in order for its function to complete
- **device_override_penalty**: The penalty issued to the agent for overriding the device when it is still on use

## Acknowledgements

This work was supported by the EMERGENT project which has received funding from the 2nd I-NERGY open call under Grant 101016508.
