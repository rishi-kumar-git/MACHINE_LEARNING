r = int(input("Rows: "))
c = int(input("Cols: "))
A = []
B = []
print("Enter matrix A:")
for i in range(r):
    A.append(list(map(int, input().split())))
print("Enter matrix B:")
for i in range(r):
    B.append(list(map(int, input().split())))
C = [[0]*c for i in range(r)]
for i in range(r):
    for j in range(c):
        for k in range(c):
            C[i][j] += A[i][k] * B[k][j]
print("Result:")
for i in C:
    print(i)
