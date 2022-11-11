import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import WordleEnvironment
from agent import WordleAgent
from utils.plot import plot_values


device = "cuda" if torch.cuda.is_available() else "cpu"

we = WordleEnvironment()
vocab_size = we.vocab_size

agent = WordleAgent(device).to(device)
agent.load_state_dict(torch.load("data/models/policy_gradient_model.pt"), strict=False)
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

# create a mask to eliminate actions not in `answer_words` vocab
vocab_mask = -1000 * torch.ones(we.vocab_size, dtype=torch.float).to(device)
answer_words_set = set(we.answer_words)
for idx, word in enumerate(we.allowable_words):
    if word in answer_words_set:
        vocab_mask[idx] = 0

# hyperparameters
batch_size = 256
n_episodes = 1000000
print_freq = 5000
save_freq = 2500

losses = []
avg_episode_values = []
avg_episode_returns = []
avg_episode_advantages = []
games_so_far = 0

for episode in range(n_episodes):

    current_state = we.game_start()
    episode_values = []

    batch_states = []
    batch_values = []
    batch_logp = []

    batch_rewards = []
    game_rewards = []

    while True:

        batch_states.append(current_state.clone())

        # get actor (`logits`) and critic (`value`) outputs and sample action
        logits, value = agent(current_state.to(device))
        logits += vocab_mask
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        # take step in environment
        game_end, current_turn, next_state, reward = we.game_step(action.item())

        batch_logp.append(logp)
        episode_values.append(value.item())
        batch_values.append(value.flatten())
        game_rewards.append(reward)

        current_state = next_state.clone()

        if game_end:

            games_so_far += 1

            batch_rewards += [sum(game_rewards)] * len(game_rewards)
            current_state = we.game_start()
            game_rewards = []

            # end episode if enough turns have been played
            if len(batch_states) > batch_size:
                break

    # create tensors from batch data
    returns = torch.Tensor(batch_rewards).to(device)
    batch_logp = torch.cat(batch_logp)  # keeps grad
    batch_values = torch.cat(batch_values)  # keeps grad

    # `returns` are undiscounted since games are short
    advantage = returns - batch_values

    # `detatch` is necessary so critic is not optimized based on actor loss
    policy_loss = (-batch_logp * advantage.detach()).mean()
    value_loss = F.smooth_l1_loss(batch_values, returns)

    # backprop
    optimizer.zero_grad()
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # track training metrics
    avg_episode_return = sum(batch_rewards) / len(batch_rewards)
    avg_episode_returns.append(avg_episode_return)

    avg_advantage = torch.mean(advantage).item()
    avg_episode_advantages.append(avg_advantage)

    avg_episode_value = sum(episode_values) / len(episode_values)
    avg_episode_values.append(avg_episode_value)

    print(
        f"Episode: {episode} |",
        f"Loss: {loss.item():.3f} |",
        f"Value: {avg_episode_value:.3f} |",
        f"Advantage: {avg_advantage:.3f} |",
        f"Returns: {avg_episode_return:.3f} |",
        f"Games: {games_so_far}",
        flush=True,
    )

    if losses and episode and episode % print_freq == 0:
        plot_values(losses, "loss")
        plot_values(avg_episode_values, "values")
        plot_values(avg_episode_returns, "returns")
        plot_values(avg_episode_advantages, "advantages")

    if (episode and episode % save_freq == 0) or episode == n_episodes - 1:
        torch.save(agent.state_dict(), "data/models/policy_gradient_model.pt")

