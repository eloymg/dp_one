N=5
fh = open("fall11_urls.txt")
for i in range(0,N):
    print(fh.readline().split()[1])
print("------------------")
for i in range(0,N):
    print(fh.readline().split()[1])
