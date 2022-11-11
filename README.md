# a2c-wordle
Playing Wordle with an advantage actor critic (A2C) policy gradient algorithm.

## Intro
This was a personal project so I could learn, understand and implement a policy gradient algorithm. I decided on [Wordle](https://www.nytimes.com/games/wordle/index.html) as the environment, since I play it frequently and it is a relatively simple game. I tried many different approaches to modelling the Wordle state, multiple agent architectures and alternative reward schemes. My final implementation follows the [approach](https://wandb.ai/andrewkho/wordle-solver/reports/Solving-Wordle-with-Reinforcement-Learning--VmlldzoxNTUzOTc4) by Andrew Ho closely.

Overall the agent has a success rate of 89.76% with an average solve rate of 3.96 turns. While this isn't human-level performance, and other, non-RL approaches exist that perform substantially better, the results were good enough for me to be confident I had learned enough to implement a reasonably successful policy gradient-based RL method (which was the real goal of this project).

## The environment
The Wordle state is stored in a 157 dimensional vector (26 characters * 6 entries + 1 turn entry). Each character in the alphabet is allocated 6 entries, the first 5 corresponding to information about the 5 columns on the Wordle board, and the last entry indicating whether the character had been explored yet (i.e. included in a word). The final entry in the state vector corresponds to the turn number (indexed from 0). Below is a diagram illustrating the state representation.

![a2c-wordle-state-vector](https://user-images.githubusercontent.com/25830706/201429630-039ab164-8f0e-4528-97ff-15e1c34486ed.png)

A turn corresponds to a word guess from the agent. The state was updated each turn to reflect the allowable positions a character could still be occupy. In the diagram above, for example, the character **B** is correct (green) in the first position, denoted by a 1. This disallows all other characters from being present in that position so their position 1 states are updated to be -1. It is possible for **B** to be present in other positions as well as position 1 however, so those positions remain at 0 to indicate an unknown state. **Z** has been attempted and is not in the word (grey), so it is inadmissable in all positions.

The reward scheme is as follows:
- -3 for each turn to encourage taking fewer turns
- 10 for predicting the correct word
- -10 for failing the game (predicting the wrong word on the final turn)

These rewards are scaled by 1/50 to avoid large gradients.

## The agent
TODO
