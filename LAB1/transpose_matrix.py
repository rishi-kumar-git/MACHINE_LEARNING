print("to find transpose od matrix")
r = int(input("row number :"))
c = int(input("col number :"))
m = []
t = []
for i in range(r):
    m.append(input().split(','))
for j in range(c):
    row = []
    for i in range(r):
        row.append(m[i][j])
    t.append(row)
for i in t:
    print(i)
