import torch
from torch import Tensor
from typing import Tuple


class ReplayBuffer:
    def __init__(self, size: int, success_buffer_portion: float, device: str):
        self.size = size
        self.device = device
        self.success_buffer_size = int(self.size * success_buffer_portion)
        self.reset_buffer()

    def reset_buffer(self):
        self.game_end_buffer = torch.zeros(self.size, dtype=torch.long).to(self.device)
        self.curr_state_buffer = torch.zeros(self.size, 104, 6, 5).to(self.device)
        self.action_buffer = torch.zeros(self.size, dtype=torch.long).to(self.device)
        self.reward_buffer = torch.zeros(self.size).to(self.device)
        self.next_state_buffer = torch.zeros(self.size, 104, 6, 5).to(self.device)
        self.n_failed_words_buffer = torch.zeros(self.size).to(self.device)

    def sample(self, sample_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        rand_idx = torch.randperm(self.size)[:sample_size].to(self.device)

        game_end_sample = self.game_end_buffer[rand_idx]
        curr_state_sample = self.curr_state_buffer[rand_idx]
        action_sample = self.action_buffer[rand_idx]
        reward_sample = self.reward_buffer[rand_idx]
        next_state_sample = self.next_state_buffer[rand_idx]
        n_failed_words_sample = self.n_failed_words_buffer[rand_idx]

        return (
            game_end_sample,
            curr_state_sample,
            action_sample,
            reward_sample,
            next_state_sample,
            n_failed_words_sample,
        )

    def add(
        self,
        game_end: int,
        curr_turn: int,
        n_failed_words: int,
        curr_state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
    ):
        # if success, add to reserved part of buffer
        if game_end and curr_turn < 6:
            # NOTE: `curr_turn` is incremented by the environment after a successful
            # or failed game, so indexing starts from 1
            rand_idx = torch.randint(0, self.success_buffer_size, (1,)).item()

        else:
            # select a random insertion index
            rand_idx = torch.randint(self.success_buffer_size, self.size, (1,)).item()

        self.game_end_buffer[rand_idx] = game_end
        self.curr_state_buffer[rand_idx] = curr_state
        self.action_buffer[rand_idx] = action
        self.reward_buffer[rand_idx] = reward
        self.next_state_buffer[rand_idx] = next_state
        self.n_failed_words_buffer[rand_idx] = n_failed_words

