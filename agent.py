import torch
from torch import nn
from pathlib import Path


class WordleAgent(nn.Module):
    def __init__(self, device: str):

        super(WordleAgent, self).__init__()

        self.device = device

        self.linear_1 = nn.Linear(157, 256)
        self.linear_2 = nn.Linear(256, 64)

        self.critic_head = nn.Linear(64, 1)
        self.actor_head = nn.Linear(64, 32)

        self.encoded_words = torch.t(torch.load(Path("data/autoencoded_words.pt"))).to(
            device
        )

        self.word_freqs = torch.load(Path("data/word_frequencies.pt")).to(device)
        self.freq_scale = 5

        self.relu = nn.ReLU()

        self.reset_action_buffer()

    def reset_action_buffer(self):
        """Clears previous actions.
        """
        self._previous_actions = []

    def update_action_buffer(self, action):
        """Keeps track of previous actions.
        """
        self._previous_actions.append(action)

    def action_mask(self, vocab_size: int):
        """Creates a mask to remove previously used actions.
        """
        mask = torch.zeros(vocab_size)
        for action in self._previous_actions:
            mask[action] = -1000
        mask = mask.to(self.device)
        return mask

    def forward(self, states: torch.Tensor):

        x = states
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        x = self.linear_1(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.relu(x)

        value = self.critic_head(x)
        logits = self.actor_head(x) @ self.encoded_words

        # shift `logits` based on word frequencies
        logits += self.freq_scale * self.word_freqs

        return logits, value
