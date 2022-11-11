from constants import Constants

import random
from typing import Optional, Tuple
from pathlib import Path
from collections import Counter
from utils.plot import plot_board
import torch


class WordleEnvironment:
    def __init__(self):
        """Instantiate the environment.
        Data obtained from https://github.com/Kinkelin/WordleCompetition
        """

        with Path("data/allowable_words.txt").open("r") as f:
            allowable_words = f.read().split("\n")

        with Path("data/answer_words.txt").open("r") as f:
            answer_words = f.read().split("\n")

        self._allowable_words = sorted(allowable_words + answer_words)
        self._vocab_size = len(self._allowable_words)
        self._allowable_words_idx_mapper = {
            idx: word for idx, word in enumerate(self._allowable_words)
        }
        self._answer_words = answer_words
        self.sampled_idxs = torch.LongTensor(list(range(len(self._allowable_words))))

        self._state: torch.Tensor
        self._reset_state()
        self._current_row = 0
        self._current_word = ""
        self._reward = 0
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        self._char_mapper = {char: idx for idx, char in enumerate(alphabet)}
        self._reverse_char_mapper = {idx: char for idx, char in enumerate(alphabet)}
        self._answer_word: str
        self._answer_word_chars: Counter
        
    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def state(self):
        return self._state.clone()

    @property
    def answer_words(self):
        return self._answer_words

    @property
    def allowable_words(self):
        return self._allowable_words

    def _reset_state(self):
        """Resets board state.
        """
        self._state = torch.zeros(157)
        self._state_plot = torch.zeros(78, 6, 5)

    def _update_state_plot(self, pred_word: str):
        """Encodes `word` into a tensor for plotting. This must be tracked
        separately from the `_state` tensor since `_state` does not contain
        necessary data to plot the board.
        """

        encodings = []
        answer_word_chars = self._answer_word_chars.copy()

        for pred_char, target_char in zip(pred_word, self._answer_word):

            encoding = torch.zeros(26, 3)

            # correct letter and position (green)
            if pred_char == target_char:
                encoding[self._char_mapper[pred_char], 0] = 1
                answer_word_chars.subtract(pred_char)

            encodings.append(encoding)

        for i, (pred_char, target_char) in enumerate(zip(pred_word, self._answer_word)):

            encoding = encodings[i]

            # correct letter but wrong position (yellow)
            if pred_char in answer_word_chars and answer_word_chars[pred_char] > 0:
                encoding[self._char_mapper[pred_char], 1] = 1
                answer_word_chars.subtract(pred_char)

            # incorrect letter (grey)
            else:
                encoding[self._char_mapper[pred_char], 2] = 1

        encodings = [encoding.flatten() for encoding in encodings]
        encodings = torch.stack(encodings, dim=-1)
        self._state_plot[:, self._current_row] = encodings

    def _update_state(self, pred_word: str):
        """Encodes `word` and updates the `_state` tensor with the encoding.
        """

        # update the plotting state
        self._update_state_plot(pred_word)

        encodings = self.state
        answer_word_chars = self._answer_word_chars.copy()

        # Iterate over full word to identify exact matches first
        for position, (pred_char, target_char) in enumerate(
            zip(pred_word, self._answer_word)
        ):
            char_idx = self._char_mapper[pred_char]

            # correct letter and position (green)
            if pred_char == target_char:

                encodings[6 * char_idx + position] = 1
                encodings[6 * char_idx + 5] = 1  # character is in word
                answer_word_chars.subtract(pred_char)

                # set maybes for this position to -1 for all other characters
                for idx in range(26):
                    if idx != char_idx:
                        encodings[6 * idx + position] = -1

        for position, (pred_char, target_char) in enumerate(
            zip(pred_word, self._answer_word)
        ):

            char_idx = self._char_mapper[pred_char]
            # encoding = encodings[5 * char_idx : 5 * (char_idx + 1)]

            # correct letter but wrong position (yellow)
            if pred_char in answer_word_chars and answer_word_chars[pred_char] > 0:

                # set "maybe" for this character to -1 for `position`
                encodings[6 * char_idx + position] = -1
                encodings[6 * char_idx + 5] = 1  # character is in word
                answer_word_chars.subtract(pred_char)

            # incorrect letter (grey)
            elif pred_char not in answer_word_chars:

                # check if character is in word
                if encodings[6 * char_idx + 5] == 1:
                    encodings[6 * char_idx + position] = -1

                # set all positions to -1 if character is not in word
                else:
                    encodings[6 * char_idx : 6 * (char_idx + 1)] = -1

        self._current_row += 1
        encodings[-1] = self._current_row

        self._state = encodings
        self._current_word = ""

    def game_start(self, start_word: Optional[str] = None):

        self._current_row = 0
        self._current_word = ""
        self._reward = 0
        self._reset_state()

        if start_word:
            self._answer_word = start_word
        else:
            self._answer_word = random.sample(self._answer_words, 1)[0]
        self._answer_word_chars = Counter(self._answer_word)

        return self.state

    def game_step(self, word_idx: int) -> Tuple[int, int, torch.Tensor, float]:
        """Adds a word to the Wordle board. Handles game logic and returns a tuple
        with the following information: (
            int: whether the game is complete or not
            int: current turn
            torch.Tensor: tensor containing the state of the Wordle board
            float: reward for step
        )
        """

        word = self._allowable_words_idx_mapper[word_idx]
        self._current_word = word

        # update state with `_current_word`
        self._update_state(word)

        # end game if word is correct
        # print(self._current_word, self._answer_word)
        if word == self._answer_word:
            return (
                1,
                self._current_row,
                self.state,
                Constants.CORRECT_WORD_REWARD,
            )

        # allowable but incorrect word on last row is a failure
        if self._current_row > 5:
            return (
                1,
                self._current_row,
                self.state,
                Constants.FAILED_GAME_REWARD,
            )

        return (
            0,
            self._current_row,
            self.state,
            Constants.NEW_ROW_REWARD,
        )

    def plot_board(self):
        """Plots the game board.
        """
        plot_board(self._state_plot)
