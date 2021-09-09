import matplotlib.pyplot as plt
import random

# 统计叶子结点个数
def count_leaf_nodes(ind):
    ind_str = str(ind)
    leaf_nodes = ind_str.count("f")
    return leaf_nodes


# 统计选择出来的特征
def count_selected_feat(ind, pset):
    list = []
    ind_str = str(ind)
    for f in reversed(pset.arguments):
        if ind_str.find(f) != -1:
            list.append(f)
            ind_str = ind_str.replace(f, '')
    list.reverse()
    return list


# 生成二维空列表
def init_two_dimensional_list(rows):
    list = []
    for row in range(rows):
        list.append([])
    return list


def graph(list, file_name):
    x = []
    y = []
    for acc, comp in list:
        x.append(acc)
        y.append(comp)
    plt.plot(x, y, linestyle="-", c="r", marker=">", alpha=0.5, lw=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # plt.title(file_name + '-pareto前沿')
    plt.xlabel('-准确率', fontsize=16)
    plt.ylabel('特征数', fontsize=16)
    # plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=10)  # 设置坐标轴刻度标记的大小
    # plt.tick_params(axis='x', which='major', labelsize=10)  # 设置坐标轴刻度标记的大小
    # plt.tick_params(axis='y', which='major', labelsize=10)  # 设置坐标轴刻度标记的大小
    plt.grid()
    plt.show()


def graph_inviduals(inviduals, file_name):
    x = []
    y = []
    for i, ind in enumerate(inviduals):
        x.append(ind.fitness.values[0])
        y.append(ind.fitness.values[1])
    plt.plot(x, y, linestyle="-", c="r", marker=">", alpha=0.5, lw=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.title(file_name + "(all fronts)")
    plt.xlabel('-准确率', fontsize=16 )
    plt.ylabel('特征数' , fontsize=16)
    # plt.tight_layout()
    plt.grid()
    plt.show()


# 去重
def distinct(pareto_first_front):
    pareto_first_front_str = map(str, pareto_first_front)
    pareto_first_front_str = set(pareto_first_front_str)
    # for ind in pareto_first_front_str:
    #     print(ind)
    pareto_no_repeat = []
    for ind in pareto_first_front:
        if str(ind) in pareto_first_front_str:
            pareto_no_repeat.append(ind)
            pareto_first_front_str.remove(str(ind))
    return pareto_no_repeat
