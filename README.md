# InfoPG: Mutual Information Maximizing Policy Gradient

This repo provides the full implementation for the paper "**Iterated Reasoning with Mutual Information in Cooperative and Byzantine Decentralized Teaming**" at the International Conference on Learning Representations (ICLR) 2022

<u>Authors</u>: Sachin Konan\*, Esmaeil Seraj\*, Matthew Gombolay

\* Co-first authors. These authors contributed equally to this work.

<u>Full Read (arXiv)</u>: `https://arxiv.org/pdf/2201.08484.pdf`

### Installation Instructions:
1. Download Anaconda
2. `conda env create --file marl.yml`
3. `cd PettingZoo`
4. `conda activate marl`
5. `python setup.py install`
6. Follow Starcraft MultiAgent Challenge Instructions Here: `https://github.com/oxwhirl/smac`

### Run PistonBall:
1. `cd pistonball`
2. To Execute Experiments:
    1. MOA: `python test_piston_ball.py -method moa`
    2. InfoPG: `python test_piston_ball.py -method infopg -k [K_LEVELS]`
    3. Adv. InfoPG: `python test_piston_ball.py -method infopg_adv -k [K_LEVELS]`
    4. Consensus Update: `python test_piston_ball.py -method consensus`
    5. Standard A2C: `python test_piston_ball.py -method a2c`
3. To Execute PR2-AC Experiments:
   1. cd `../pr2-ac/pistonball/`
   2. `python distributed_pistonabll_train.py -batch 4 -workers [NUM CPUS]`
   3. Results will be saved in `experiments/pistonball/[DATETIME OF RUN]/`
#### Run Fraud (Byzantine Experiments):
1. MOA: `python batch_pistoncase_moa_env.py`
2. InfoPG: `python batch_pistoncase_infopg_env.py`

### Run Pong:
1. `cd pong`
2. To Execute MOA Experiments:
   1. `cd pong_moa`
   2. MOA: `python distributed_pong_moa_train.py -batch 16 -workers [NUM CPUS]`
   3. Results will be saved in `experiments/pong/[DATETIME OF RUN]/`
3. To Execute PR2-AC Experiments:
   1. cd `../pr2-ac/pong/`
   2. `python distributed_pong_train.py -batch 16 -workers [NUM CPUS]`
   3. Results will be saved in `experiments/pong/[DATETIME OF RUN]/`
4. To Execute Other Experiments:
   1. InfoPG: `python distributed_pong_train.py -batch 16 -workers [NUM CPUS] -k [K_LEVELS] -adv info -critic`
   2. Adv. InfoPG: `python distributed_pong_train.py -batch 16 -workers [NUM CPUS] -k [K_LEVELS] -adv normal`
   3. Consensus Update: `python distributed_pong_train.py -batch 16 -workers [NUM CPUS] -k 0 -adv normal -consensus`
   4. Standard A2C: `python distributed_pong_train.py -batch 16 -workers [NUM CPUS] -k 0 -adv normal`
   5. Results will be saved in `experiments/pong/[DATETIME OF RUN]/`

### Run Walker:
1. `cd walker`
2. To Execute MOA Experiments:
  1. `cd walker_moa`
  2. MOA: `python distributed_walker_train_moa.py -batch 16 -workers [NUM CPUS]`
  3. Results will be saved in `experiments/walker_moa/[DATETIME OF RUN]/`
3. To Execute PR2-AC Experiments:
   1. cd `../pr2-ac/walker/`
   2. `python distributed_walker_train.py -batch 16 -workers [NUM CPUS]`
   3. Results will be saved in `experiments/walker/[DATETIME OF RUN]/`
4. To Execute Other Experiments:
   1. InfoPG: `python distributed_walker_train.py -batch 16 -workers [NUM CPUS] -k [K_LEVELS] -adv info -critic`
   2. Adv. InfoPG: `python distributed_walker_train.py -batch 16 -workers [NUM CPUS] -k [K_LEVELS] -adv normal`
   3. Consensus Update: `python distributed_walker_train.py -batch 16 -workers [NUM CPUS] -k 0 -adv normal -consensus`
   4. Standard A2C: `python distributed_walker_train.py -batch 16 -workers [NUM CPUS] -k 0 -adv normal`
   5. Results will be saved in `experiments/walker/[DATETIME OF RUN]/`


### Run Starcraft:
1. `cd starcraft`
2. To Execute MOA Experiments:
   1. `cd moa`
   2. MOA: `python distributed_starcraft_train_moa.py -batch 128 -workers [NUM CPUS] -positive_rewards`
   3. Results will be saved in `experiments/starcraft/[DATETIME OF RUN]/`
3. To Execute PR2-AC Experiments:
   1. cd `../pr2-ac/starcraft/`
   2. `python distributed_starcraft_train.py -batch 128 -workers [NUM CPUS]`
   3. Results will be saved in `experiments/starcraft/[DATETIME OF RUN]/`
4. To Execute Other Experiments:
   1. InfoPG: `python distributed_walker_train.py -batch 128 -workers [NUM CPUS] -k [K_LEVELS] -adv info -critic -positive_rewards`
   2. Adv. InfoPG: `python distributed_walker_train.py -batch 128 -workers [NUM CPUS] -k [K_LEVELS] -adv normal -positive_rewards`
   3. Consensus Update: `python distributed_walker_train.py -batch 128 -workers [NUM CPUS] -k 0 -adv normal -consensus -positive_rewards`
   4. Standard A2C: `python distributed_walker_train.py -batch 128 -workers [NUM CPUS] -k 0 -adv normal -positive_rewards`
   5. Results will be saved in `experiments/starcraft/[DATETIME OF RUN]/`

   

   



