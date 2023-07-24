import re

from graph import Graph


class Automata:
    def __init__(self, state_transfer_table, state=0):
        self.state_transfer_table = state_transfer_table
        self.state = 0
        self.workspace = ""

    def step(self):
        available = self.state_transfer_table[self.state]
        for condition, next_state in available.items():
            if "*" in condition:
                condition = condition.replace("*", ".")
            if re.match(f"^{condition}", condition):
                self.state = next_state
                self.workspace = re.sub(f"^{condition}", "", self.workspace)
                return next_state
        return -1
