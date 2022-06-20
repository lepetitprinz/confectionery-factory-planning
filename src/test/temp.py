def seq(n):
    val = [1, 2]
    if n >= 2:
        for i in range(2, n):
            val.append((val[i - 1] + val[i - 2]) % 15746)

        return val[n-1]

n = int(input())
print(seq(n) % 15746)

