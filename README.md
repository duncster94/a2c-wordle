# a2c-wordle
Playing Wordle with an advantage actor critic (A2C) policy gradient algorithm.

## Intro
This was a personal project so I could learn, understand and implement a policy gradient algorithm. I decided on [Wordle](https://www.nytimes.com/games/wordle/index.html) as the environment, since I play it frequently and it is a relatively simple game. I tried many different approaches to modelling the Wordle state, multiple agent architectures and alternative reward schemes. My final implementation follows the [approach](https://wandb.ai/andrewkho/wordle-solver/reports/Solving-Wordle-with-Reinforcement-Learning--VmlldzoxNTUzOTc4) by Andrew Ho closely.

Overall the agent has a success rate of 89.76% with an average solve rate of 3.96 turns. While this isn't human-level performance, and other, non-RL approaches exist that perform substantially better, the results were good enough for me to be confident I had learned enough to implement a reasonably successful policy gradient-based RL method (which was the real goal of this project). If you have any questions about this project feel free to open an [issue](https://github.com/duncster94/a2c-wordle/issues).

## The environment
The Wordle state is stored in a 157 dimensional vector (26 characters * 6 entries + 1 turn entry). Each character in the alphabet is allocated 6 entries, the first 5 corresponding to information about the 5 columns on the Wordle board, and the last entry indicating whether the character had been explored yet (i.e. included in a word). The final entry in the state vector corresponds to the turn number (indexed from 0). Below is a diagram illustrating the state representation.

![a2c-wordle-state-vector](https://user-images.githubusercontent.com/25830706/201429630-039ab164-8f0e-4528-97ff-15e1c34486ed.png)

A turn corresponds to a word guess from the agent. The state was updated each turn to reflect the allowable positions a character could still occupy. In the diagram above, for example, the character **B** is correct (green) in the first position, denoted by a 1. This disallows all other characters from being present in that position so their position 1 states are updated to be -1. It is possible for **B** to be present in other positions as well as position 1 however, so those positions remain at 0 to indicate an unknown state. **Z** has been attempted and is not in the word (grey), so it is inadmissable in all positions.

The reward scheme is as follows:
- -3 for each turn to encourage taking fewer turns
- 10 for predicting the correct word
- -10 for failing the game (predicting the wrong word on the final turn)

These rewards are scaled by 1/50 to avoid large gradients.

## The agent
For the agent I used the advantage actor critic (A2C) algorithm. This is similar to the vanilla REINFORCE policy gradient approach, but with an added state value estimate used as a variance reducing baseline. While REINFORCE uses (un)discounted returns to update the policy distribution, A2C subtracts the state value estimate from the returns to yield the **advantage** - a measure of how much better or worse the return given to an action was than the expected return of being in that state.

The agent is a simple MLP with a policy head and a value head. I used an autoencoder to generate 32 dimensional embeddings of the Wordle vocabulary and computed logits over each word via a dot product with the policy head. The agent includes an action buffer to track which words it has used in the current Wordle game, and generate a logits mask to prevent these words from being used again. This was done because using the same word multiple times is a strictly bad action as it provides no additional information and advances turns.

## Training
I trained the Wordle agent for ~17 million games before evaluating its final performance. The agent converged on a strategy of starting with the word "SANER" followed by "BITTY" if all characters in "SANER" were grey. "BITTY" seems suboptimal considering two T's are being used, but perhaps the agent knows something I don't :smile:.

There were a number of notable failure cases though where the agent made mistakes an experienced human player would be unlikely to make. For example, the word "TYING":

<p align="center">
<img src="https://user-images.githubusercontent.com/25830706/201995461-6f141b27-286a-4af5-a480-aeb1c8241e56.png">
</p>

After "POINT", it's revealed that the Y and T are present in the first two positions, however the agent predicts "THING" before correctly predicting "TYING". Given that "Y" in the second position is quite rare, it's likely the agent saw few examples of this while training, resulting in suboptimal performance.

## Conclusion
This was a very instructional project. Unexpectedly, I found that designing the Wordle environment - with an accurate state representation and reward scheme - was a greater challenge than building the agent itself. Despite playing with reasonable proficiency, it's clear to me the agent does not yet fully understand the rules of the environment. I wonder if there are informative inductive biases that could be incorporated into the agent architecture such that letter frequency does not influence actions when certain letters are known to be present in the answer word. This would rectify situations like "TYING" mentioned above, where Y was known to be in the answer word but was not used.
