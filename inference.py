import torch

from agent import WordleAgent
from environment import WordleEnvironment

device = "cuda" if torch.cuda.is_available() else "cpu"

we = WordleEnvironment()
answer_words = we.answer_words

agent = WordleAgent(device).to(device)
agent.load_state_dict(torch.load("data/models/policy_gradient_model.pt"))

# create a mask to eliminate actions not in `answer_words` vocab
vocab_mask = -1000 * torch.ones(we.vocab_size, dtype=torch.float).to(device)
answer_words_set = set(we.answer_words)
for idx, word in enumerate(we.allowable_words):
    if word in answer_words_set:
        vocab_mask[idx] = 0

successes = 0
turn_counts = []

for answer_word in answer_words:
    turns = 0
    state = we.game_start(answer_word)
    agent.reset_action_buffer()

    while True:
        mask = agent.action_mask(we.vocab_size) + vocab_mask
        logits, _ = agent(state.to(device))
        action = torch.argmax(logits.flatten() + mask).item()
        agent.update_action_buffer(action)

        done, turn, state, _ = we.game_step(action)

        if done:
            current_word = we._allowable_words_idx_mapper[action]
            if current_word == answer_word:
                successes += 1
                turn_counts.append(turn)
            we.plot_board()
            break

success_rate = 100 * (successes / len(answer_words))
avg_turns = sum(turn_counts) / len(turn_counts)
print(f"Success rate: {success_rate:.2f}% | Avg turns: {avg_turns:.2f}")
# Success rate: 89.76% | Avg turns: 3.96
