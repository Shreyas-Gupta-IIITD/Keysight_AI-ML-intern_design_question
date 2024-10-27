# Keysight AI-ML intern Design Questions
Repository for solutions to Keysight Design questions for AI/ML intern position.

Submitted by: Shreyas Gupta, Akshet Patial

## Design Problem Statement 1: 
Build RL solution which initially has a bad policy,so learn the correct policy over time based on the correct/incorrect actions taken by agent.

### Proposed Solution 1:
Play the Pac-Man game using Deep Q-Network(DQN).

GIF of the agent playing `MS PacMan` game environment using Gym:

![Agent Playing](assets/pacman.gif)


Succefully run the DQN for 3 episodes which take 2 hr. The complete execution of DQN on 500 episodes and 2000 steps is expected to take 300+ hours.

![Agent Playing](assets/episode.png)

To develop a fully working solution for the given Problem statement, now use a simpler algorithm on a smaller environment  (done in following soluiton 2)

### Proposed Solution 2:
Solve the Cliff Walking problem and and compare the performance using Q-Learning, SARSA, Expected-SARSA

GIF of the agent traversing in the `Cliff Walking` environment using Gym:

![Agent Playing](assets/cliffwalking.gif)


video of the agent reaching in the Goal state in `Cliff Walking` environment using the learned policy:

![Agent Playing](assets/episode548.gif)


The comparison plot for algorithms used in solution 2 is as:

![Agent Playing](assets/plot.png)

## Design Problem Statement 2:
Build a model to detect presence of Inductor (spiral shaped) using the given Test Spiral.kicad_pcb file.

### Solution:
