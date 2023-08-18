data = [
    {"id": 1, "start": 1, "end": 4},
    {"id": 2, "start": 3, "end": 5},
    {"id": 3, "start": 0, "end": 6},
    {"id": 4, "start": 5, "end": 7},
    {"id": 5, "start": 3, "end": 8},
    {"id": 6, "start": 5, "end": 9},
    {"id": 7, "start": 6, "end": 10},
    {"id": 8, "start": 8, "end": 11},
    {"id": 9, "start": 8, "end": 12},
    {"id": 10, "start": 2, "end": 13},
    {"id": 11, "start": 12, "end": 14},
]
data = list(map(lambda x: ([x["start"], x["end"]]), data))
data = [(1, 3), (2, 6), (5, 6), (7, 8), (4, 9), (6, 11), (9, 11), (10, 12), (11, 12), (11, 13)]
schedule = []
data = sorted(data, key=lambda x: x[1])
last = 0
while len(data) > 0:
    stop = False
    for interval in data:
        if interval[0] > last:
            schedule.append(interval)
            last = interval[1]
            data.remove(interval)
            stop = True
    if stop:
        break
print(data)
print(schedule)

edges = [
    ("v1", "v5", 5),
    ("v2", "v1", 7),
    ("v2", "v7", -4),
    ("v3", "v2", 5),
    ("v4", "v3", -2),
    ("v5", "v4", -4),
    ("v5", "v10", -6),
    ("v6", "v1", 8),
    ("v6", "v8", -1),
    ("v7", "v9", 4),
    ("v8", "v3", -2),
    ("v8", "v10", -2),
    ("v9", "v4", -3),
    ("v9", "v6", -7),
]
edges = list(map(lambda x: (x[0], x[1], str(x[2])), edges))
edges.sort(key=lambda x: int(x[1][1:]))
import json

for e in edges:
    print(f"({', '.join(list(e))}),")
