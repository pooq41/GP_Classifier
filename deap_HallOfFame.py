#保存topbest个体
class HallOfFame(object):

    def __init__(self, topnum ,similar=eq):
        self.items = list()
        self.similar = similar
        self.topnum = topnum
        self.topbestfit = []
        self.bestinddict = {}
        self.bestattrnum = {}

    def update(self, population):
        for ind in population:
            self.topbestfit.sort()
            attrNum = str(ind).count('f')
            a = ind.fitness.values[0] - 0.001 * attrNum

            if len(self.items) < self.topnum and a not in self.topbestfit:
                self.insert(ind)
                self.topbestfit.append(a)
                self.bestinddict[a] = ind
                self.bestattrnum[a] = attrNum
                self.topbestfit.sort()
                continue

            elif a in self.topbestfit and attrNum < self.bestattrnum[a]:
                self.items.remove(self.bestinddict[a])
                self.insert(ind)
                self.bestinddict[a] = ind
                self.bestattrnum[a] = attrNum

            elif a not in self.topbestfit and a < self.topbestfit[-1] :
                self.items.remove(self.bestinddict[self.topbestfit[-1]])
                self.insert(ind)
                del self.bestinddict[self.topbestfit[-1]]
                self.bestinddict[a] = ind
                self.bestattrnum[a] = attrNum
                self.topbestfit[-1] = a

    def insert(self, item):
        item = deepcopy(item)
        self.items.append(item)

    def clear(self):
        del self.items[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)
