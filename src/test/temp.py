def tetromino(i, j, matrix, move_list, depth=0, val_list=[]):
    visited = []

    if depth < 4:
        val = matrix[i][j]
        val_list.append(val)
        for move in move_list:
            i += move[0]
            j += move[1]
            if (0 <= i < n) and (0 <= j < m):
                val += tetromino(i, j, matrix, move_list, depth=depth + 1, val_list=val_list)

    return sum(val_list)

n, m = map(int, input().split())
move_list = [(1, 0), (0, 1), (-1, 0), (0, -1)]

matrix = []
for _ in range(n):
    matrix.append(list(map(int, input().split())))

max_val = 0
for i in range(n):
    for j in range(m):
        val_list = tetromino(i, j, matrix, move_list)
        if max(val_list) > max_val:
            max_val = max(val_list)

print(max_val)