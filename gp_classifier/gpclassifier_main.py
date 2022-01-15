from gpclassifier import gp_classifier, count_leaf_nodes
from a_others.read_data import RD
import random
import numpy as np

dir_name = "../datasets/arff"
# GSES = ['GSE42408', 'GSE46205', 'GSE76613', 'GSE145709', 'GSE14728']  # datasets we used
GSES = ['GSE145709']  # dataset runningza
resultspath = '../JILU/1_0.01/'  # experiment result write path
cishu = 30  # number of program runs


def main():
    for GSE in GSES:
        # Load the dataset and initialize the variables
        rd = RD()
        all_datas = rd.read_arff(dir_name, GSE)
        all_datas = rd.mms(all_datas)

        feat_num = len(all_datas[0]) - 1
        instanceNum = len(all_datas)

        acc_sum = 0.0
        feat_sum = 0
        selected_feat_all = []
        selected_feat_num_all = []
        acc_all = []

        for i in range(cishu):
            # Get training set and testing set
            data_training = random.sample(all_datas, int(len(all_datas) * 0.7))
            # data_training = random.sample(all_datas, int(len(all_datas)))
            data_testing = [i for i in all_datas if i not in data_training]

            # training
            acc, hof, evalf, pset = gp_classifier(data_training, data_testing, feat_num, instanceNum)
            # print("最佳个体树：", len(hof.items[0]), hof.items[0])

            # testing
            print("第%d次分类准确率为" % i, acc)
            selected_feat = []
            for item in hof.items:
                selected_feat.append(str(item))
                # print(count_leaf_nodes(item))
                feat_sum += count_leaf_nodes(item)
            selected_feat_all.append(selected_feat)
            # selected_feat_num = len(selected_feat)
            # selected_feat_num_all.append(selected_feat_num)
            acc_all.append(acc)
            acc_sum += acc
        acc_avg = acc_sum / cishu
        selected_feat_num_avg = feat_sum / cishu

        # Write experimental results
        with open(resultspath + GSE + '.txt', 'w') as f:

            f.write('样本数：' + str(len(all_datas)) + '\t原始特征数：' + str(feat_num) + '\n')
            f.write("平均准确率:" + str(acc_avg) + '\n最大值:' + str(max(acc_all)) + '\t最小值：' + str(min(acc_all)) +
                    '\t标准差:' + str(np.std(acc_all)) + "平均特征数:" + str(selected_feat_num_avg) + '\n')
            f.write('\n')
            for item in range(len(selected_feat_all)):
                f.write('第%d次：\n' % item)
                for i in selected_feat_all[item]:
                    f.write(str(i) + '\n')
                f.write('\n')


if __name__ == '__main__':
    main()
