import operator
import itertools
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from a_others.parameters import *
from a_others.func_tools import *


# 统计叶子结点个数
def count_leaf_nodes(ind):
    ind_str = str(ind)
    leaf_nodes = ind_str.count("f")
    return leaf_nodes


#  统计选择出来的特征
def count_selected_feat(ind, pset):
    list = []
    ind_str = str(ind)
    for f in reversed(pset.arguments):
        if ind_str.find(f) != -1:
            list.append(f)
            ind_str = ind_str.replace(f, '')
    list.reverse()
    return list


def gp_classifier(data_training, data_testing, feat_num, instanceNum):
    # defined a new primitive set for strongly typed GP
    #  创建一个迭代器，它返回指定次数的对象。如果未指定，则无限返回对象。
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, feat_num), float, "f")

    # print("pset.arguments:", pset.arguments)

    # boolean operators
    # pset.addPrimitive(operator.and_, [bool, bool], bool)
    # pset.addPrimitive(operator.or_, [bool, bool], bool)
    # pset.addPrimitive(operator.not_, [bool], bool)    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    def Div(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(Div, [float, float], float)
    # pset.addPrimitive(operator.neg, [float], float)
    #
    # pset.addPrimitive(operator.lt, [float, float], float)
    # pset.addPrimitive(operator.gt, [float, float], float)

    # pset.addTerminal(random.random(), float)
    # pset.addTerminal(random.random()-1, float)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def classes(func, datas):
        if func(datas[:-1]) >= 0:
            return 1.0
        else:
            return 0.0

    def evalfunc(individual, all_datas):
        if len(all_datas) == int(instanceNum * 0.7):
        # if len(all_datas) == int(instanceNum):
            attrNum = count_leaf_nodes(individual)
            func = toolbox.compile(expr=individual)
            list = []
            for datas in all_datas:
                result = True if classes(func, datas) == datas[-1] else False
                list.append(result)
            acc = sum(list) / len(list)
            k = 0.001
            # return 1-acc,
            return (1 - k) * (1 - acc) + k * attrNum,
            # return (1 - k) * (1 - acc) + k * len(individual),
        else:
            func = toolbox.compile(expr=individual)
            list = []
            for datas in all_datas:
                result = True if classes(func, datas) == datas[-1] else False
                list.append(result)
            acc = sum(list) / len(list)
            return acc,

    toolbox.register("evaluate", evalfunc, all_datas=data_training)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 初始化种群
    # random.seed(seed)
    pop = toolbox.population(n=N_POP)

    hof = tools.HallOfFame(11)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("平均值", numpy.mean)
    stats.register("标准差", numpy.std)
    stats.register("最小值", numpy.min)
    stats.register("最大值", numpy.max)
    # print("统计器的key值：", stats.key)
    # print("统计器注册的函数：", stats.functions.items())
    # print(stats.fields)

    # 装饰器
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    # 开始进化
    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, N_GEN, stats, halloffame=hof, verbose=True)
    print("-" * 30 + "进化完成" + "-" * 30)

    print(str(len(hof.items)) + '个最优个体')
    for item in hof.items:
        print(item.fitness, item)
    acc = 0

    if len(hof.items) == 1:
        acc = evalfunc(hof.items[0], data_testing)[0]

    else:
        predict = []
        for data in data_testing:
            preacc = []
            for item in hof.items:
                func = toolbox.compile(expr=item)
                preacc.append(classes(func, data))
            result = 1.0 if (sum(preacc) > len(preacc) / 2) else 0.0
            predict.append(1 if (result == data[-1]) else 0)
        acc = sum(predict) / len(predict)

    return acc, hof, evalfunc, pset
