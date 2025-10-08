from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CliffWalking:
    def __init__(self, cols = 12, rows = 4):
        self.cols = cols
        self.rows = rows
        self.f = self._create_cached_table()

    def __call__(self, state: tuple[int, int], action: Action):
        return self.f[state][action]

    def _create_cached_table(self):
        return {
            (x, y): {action: {
                "p_trans": 0,
                "next_state": (0, 0)
            } for action in Action} for x in range(self.cols) for y in range(self.rows)
        }