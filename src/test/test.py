n = int(input())
cnt = 0
positions = []

def check_loc(n, i, positions):
    row = list(range(n))
    for x, y in positions:
        if x in row:
            row.remove(x)
        for loc in make_diagonal(n, i, (x, y)):
            if loc in row:
                row.remove(loc)

    return row

def make_diagonal(n, i, coordinate):
    location = []
    for add in [i, -i]:
        if 0 <= coordinate + add < n:
            location.append(coordinate + add)

    return location
