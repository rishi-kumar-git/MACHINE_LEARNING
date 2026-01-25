import random
l = []
for i in range(100):
    l.append(random.randint(100,150))
l.sort()
mean = sum(l)/100
med = (l[49] + l[50]) / 2
mode = l[0]
for i in l:
    if l.count(i) > l.count(mode):
        mode = i
print("Mean:", mean)
print("Median:", med)
print("Mode:", mode)
