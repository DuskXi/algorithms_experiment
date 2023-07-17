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
