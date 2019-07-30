# Bandits algorithms

For this practical work in my Reinforcement Learning class, we had to implement a few bandit algorithms, namely:

* Epsilon-greedy bandit (https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
* Besa https://hal.archives-ouvertes.fr/hal-01025651v1/document
* UCB1 https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
* Softmax https://www.cs.mcgill.ca/~vkules/bandits.pdf
* KL-UCB https://hal.archives-ouvertes.fr/hal-00738209v2

The implementations can be found in `agent.py`

To run the experiment, use `python main.py --niter 1000 --batch 2000`
