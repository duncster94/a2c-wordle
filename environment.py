from constants import Constants

import random
from typing import Optional, Tuple
from pathlib import Path
from collections import Counter
from utils.plot_board import plot_board
import torch


class WordleEnvironment():

    def __init__(self):
        """Instantiate the environment.
        Data obtained from https://github.com/Kinkelin/WordleCompetition
        """

        with Path("data/allowable_words.txt").open("r") as f:
            allowable_words = f.read().split("\n")

        with Path("data/answer_words.txt").open("r") as f:
            answer_words = f.read().split("\n")

        # Cast to set to allow O(1) comparisons during the game
        self._allowable_words = set(allowable_words + answer_words)

        self._answer_words = answer_words
        self._state: torch.Tensor
        self._reset_state()
        self._current_row = 0
        self._current_col = 0
        self._current_word = ""
        self._reward = 0
        self._char_mapper = {char: idx for idx,
                             char in enumerate("abcdefghijklmnopqrstuvwxyz")}
        self._answer_word: str
        self._answer_word_chars: Counter

    @property
    def state(self):
        return self._state

    def _reset_state(self):
        """Each cell can have 26 different letters * 4 success states (correct letter 
        and position, correct letter but wrong position, wrong letter, and unknown) in 
        addition to the empty state (zeros). The Wordle board contains 6 rows and 5 
        columns, yielding a state tensor that is 6 x 5 x 26 x 4.
        """
        self._state = torch.zeros(6, 5, 26, 4)

    def _reset_turn_state(self):
        """Resets the current turn word state. Called if the agent predicts a word
        not in `_allowable_words` to clear the state for `_current_row` and start again.
        """

        self._state[self._current_row] = torch.stack([torch.zeros(26, 4)] * 5)
        self._current_col = 0
        self._current_word = ""
        self._reward += Constants.NOT_WORD_LIST_REWARD

    def _update_state_char(self, pred_char: str):
        """Encodes `pred_char` character and adds to state. No information is given
        about success. Called when the agent predicts a letter to add to the current
        word. Used to provide a history of characters to the agent.
        """

        encoding = torch.zeros(26, 4)
        encoding[self._char_mapper[pred_char], 3] = 1
        self._state[self._current_row, self._current_col] = encoding
        self._current_col += 1

    def _update_state_word(self, pred_word: str):
        """Encodes `word` and updates the `_state` tensor with the encoding.
        Called when the agent "submits" the word.
        """

        encodings = []
        answer_word_chars = self._answer_word_chars.copy()
        for pred_char, target_char in zip(pred_word, self._answer_word):

            encoding = torch.zeros(26, 4)

            # correct letter and position (green)
            if pred_char == target_char:
                encoding[self._char_mapper[pred_char], 0] = 1
                answer_word_chars.subtract(pred_char)
                self._reward += Constants.GREEN_CHAR_REWARD

            # correct letter but wrong position (yellow)
            elif pred_char in answer_word_chars and answer_word_chars[pred_char] > 0:
                encoding[self._char_mapper[pred_char], 1] = 1
                answer_word_chars.subtract(pred_char)
                self._reward += Constants.YELLOW_CHAR_REWARD

            # incorrect letter (grey)
            else:
                encoding[self._char_mapper[pred_char], 2] = 1
                self._reward += Constants.GREY_CHAR_REWARD

            encodings.append(encoding)

        encodings = torch.stack(encodings)
        self._state[self._current_row] = encodings
        self._current_row += 1
        self._current_col = 0
        self._current_word = ""
        self._reward += Constants.NEW_ROW_REWARD

    def game_start(self, debug_word: Optional[str] = None):
        self._current_row = 0
        self._current_col = 0
        self._current_word = ""
        self._reward = 0
        self._reset_state()

        if debug_word:
            self._answer_word = debug_word
        else:
            self._answer_word = random.sample(self._answer_words, 1)[0]
        self._answer_word_chars = Counter(self._answer_word)

    def game_step(self, char: str) -> Tuple[bool, torch.Tensor, float]:
        """Adds a character to the Wordle board. Handles game logic and returns a tuple
        with the following information: (
            bool: whether the game is complete or not
            torch.Tensor: tensor containing the state of the Wordle board
            float: cumulative reward
        )
        """

        assert len(char) == 1

        self._current_word += char

        # check if current input constitutes a full word
        if len(self._current_word) == 5:

            # end game if word is correct
            if self._current_word == self._answer_word:
                self._reward += Constants.CORRECT_WORD_REWARD
                self._update_state_word(self._current_word)
                return (True, self._state, self._reward)

            # ensure word is allowable
            if self._current_word in self._allowable_words:

                # allowable but incorrect word on last row is a failure
                if self._current_row == 5:
                    self._reward += Constants.FAILED_GAME_REWARD
                    return (True, self._state, self._reward)

                # if not last row, update state with `_current_word`
                self._update_state_word(self._current_word)
                return (False, self._state, self._reward)

            # otherwise reset row
            else:
                self._reset_turn_state()
                return (False, self._state, self._reward)

        # if `_current_word` is not a full word, update state with `char`
        else:
            self._update_state_char(char)
            return (False, self._state, self._reward)

    def plot_board(self):
        """Plots the game board.
        """
        plot_board(self._state)


we = WordleEnvironment()
we.game_start("crate")
print(we.game_step("r")[2])
print(we.game_step("o")[2])
print(we.game_step("v")[2])
print(we.game_step("e")[2])
print(we.game_step("r")[2])

print(we.game_step("r")[2])
print(we.game_step("r")[2])
print(we.game_step("r")[2])
print(we.game_step("r")[2])
print(we.game_step("r")[2])

print(we.game_step("c")[2])
print(we.game_step("l")[2])
print(we.game_step("o")[2])
print(we.game_step("s")[2])
print(we.game_step("e")[2])

print(we.game_step("s")[2])
print(we.game_step("o")[2])
print(we.game_step("u")[2])
print(we.game_step("n")[2])
print(we.game_step("d")[2])

print(we.game_step("p")[2])
print(we.game_step("l")[2])
print(we.game_step("a")[2])
print(we.game_step("y")[2])
print(we.game_step("s")[2])

print(we.game_step("s")[2])
print(we.game_step("h")[2])
print(we.game_step("i")[2])
print(we.game_step("f")[2])
print(we.game_step("t")[2])

print(we.game_step("c")[2])
print(we.game_step("r")[2])
print(we.game_step("a")[2])
print(we.game_step("t")[2])
print(we.game_step("e"))

we.plot_board()
