import re

from matplotlib import pyplot as plt

from graph import Graph


class Automata:
    def __init__(self, state_transfer_table, state=0):
        self.state_transfer_table = state_transfer_table
        self.state = state
        self.workspace = ""

    def step(self):
        available = self.state_transfer_table[self.state]
        for next_state, conditions in available.items():
            for condition in conditions:
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

    def draw(self):
        graph = Graph(directed=True)
        for state, transfer in self.state_transfer_table.items():
            graph.add_node(state)
        for state, transfer in self.state_transfer_table.items():
            for next_state, conditions in transfer.items():
                graph.add_edge(state, next_state, ', '.join(conditions))
        return graph.draw_as_label_weight()


def run_test():
    transfer_table = {
        "even": {
            "odd": ["a"],
            "even": ["b"]
        },
        "odd": {
            "even": ["a"],
            "odd": ["b"]
        }
    }
    transfer_table = {
        "a+b": {
            "b": ["b"],
            "a+b": ["a", "b"]
        },
        "b": {
            "b": ["b"],
        }
    }
    transfer_table = {
        "init": {
            "1s": ["1"]
        },
        "1s": {
            "1s": ["1"],
            "mid": ["0"]
        },
        "mid": {
            "end": ["1"],
            "mid": ["0"]
        },
        "end": {
            "end": ["0", "1"],
        }
    }
    automata = Automata(transfer_table)
    print(automata.run("11111001101", list(transfer_table.keys())[0]))
    draw = automata.draw()
    plt.show()


if __name__ == '__main__':
    run_test()
