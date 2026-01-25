l1 = list(map(int, input("Enter list 1: ").split(',')))
l2 = list(map(int, input("Enter list 2: ").split(',')))
cnt = 0
for i in l1:
    if i in l2:
        cnt += 1
print("Common elements :", cnt)

