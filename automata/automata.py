import re

from graph import Graph


class Automata:
    def __init__(self, state_transfer_table, state=0):
        self.state_transfer_table = state_transfer_table
        self.state = state
        self.workspace = ""

    def step(self):
        available = self.state_transfer_table[self.state]
        for condition, next_state in available.items():
            if "*" in condition:
                condition = condition.replace("*", ".")
            if re.match(f"^{condition}", self.workspace):
                self.state = next_state
                self.workspace = re.sub(f"^{condition}", "", self.workspace)
                return next_state
        self.state = -1
        return -1

    def run(self, string, init):
        self.workspace = string
        self.state = init
        transfer_history = []
        while self.state != -1:
            result = self.step()
            transfer_history.append((result, self.workspace))
        return transfer_history


def run_test():
    transfer_table = {
        "even": {
            "a": "odd",
            "b": "even"
        },
        "odd": {
            "a": "even",
            "b": "odd"
        }
    }
    automata = Automata(transfer_table)
    print(automata.run("aababa", "even"))


if __name__ == '__main__':
    run_test()
