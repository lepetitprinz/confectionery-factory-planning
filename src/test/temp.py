
n = int(input())
arr = [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]
# for _ in range(n):
#     arr.append(list(map(int, input().split())))

split = 2
for i in range(1, n):
    for j in range(split):
        if j == 0:
            arr[i][j] = arr[i][j] + arr[i - 1][j]
        elif i == j:
            arr[i][j] = arr[i][j] + arr[i - 1][j - 1]
        else:
            arr[i][j] = arr[i][j] + max(arr[i - 1][j - 1], arr[i - 1][j])
    split += 1

print(max(arr[n-1]))


