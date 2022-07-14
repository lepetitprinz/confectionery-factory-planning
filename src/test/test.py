n, m = map(int, input().split())
books = list(map(int, input().split()))
books = sorted(books, reverse=True)

cnt = 0
weight = m
while books:
    for book in books:
        if book <= weight:
            weight -= book
            books.remove(book)
            if weight == 0:
                cnt += 1
                weight = m
                break
    cnt += 1
    weight = m

print(cnt)