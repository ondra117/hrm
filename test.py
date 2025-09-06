class Test:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for d in self.data:
            yield d


t = Test(range(5))

for d in t:
    print(d)
for d in t:
    print(d)
