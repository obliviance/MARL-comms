# MARL-comms

## Setup
1. Create a python environment using `python -m venv env`.
2. Start the environment.
    1. On Windows `source env/Scripts/activate`
    2. On Linux `source env/bin/activate`
3. Install the dependencies with `pip install -r requirements.txt`

## Usage
Run the environment with `python mpe_env.py` in the terminal
Add policies in the `mpe_model.py` file

## Environment
The main environment we are investigating is the Multi Particle Environment(MPE) available through PettingZoo, which is a python library that provides multiagent environments, associated with Farama Gymnasium.

Link: https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/

We will also investigate the Drone Swarm Search environment.

Link: https://github.com/pfeinsper/drone-swarm-search

## Misc
Make sure to use the PettingZoo version of MPE and not the MPE2 package. MPE2 does not work, it's got some kind of file mismatch with PettingZoo's AgentSelector(or more subtle issue).

