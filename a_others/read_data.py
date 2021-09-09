import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


class RD(object):

    def __init__(self):
        self.one_list = []
        self.all_list = []
        # self.label_str = []
        # self.label_num_str = ""
        self.labels = []
        # self.m = 0

    def read_csv(self, dir_name, file_name):
        with open(dir_name + "/" + file_name + ".csv") as f:
            Reader = csv.reader(f)
            all_list = list(list(float(elem) for elem in row) for row in Reader)
            return all_list

    def read_arff(self, dir_name, file_name):
        # with open("../" + dir_name + "/" + file_name + ".arff", "r") as arff_file:
        with open(dir_name + "/" + file_name + ".arff", "r") as arff_file:
            for line in arff_file:
                if not (line.startswith("@") or line.startswith("%")):
                    if line != "\n":
                        line_list = line.strip("\n").split(",")
                        # self.label_str.append(line_list[-1])
                        features = line_list[:-1]
                        label = line_list[-1]
                        # print("features:", features)
                        # print("label:", label)
                        self.labels.append(label)
                        self.labels = sorted(set(self.labels), key=self.labels.index)
                        # print("labels:", self.labels)
                        for index, lab in enumerate(self.labels):
                            if label == lab:
                                label = str(index)
                                break
                        features.append(label)
                        self.one_list = features
                        # print(self.list)
                        # break

                        self.one_list = [float(item) if item!="?" else 0.0 for item in self.one_list]
                        self.all_list.append(self.one_list)

                        # self.m += 1
                        # if self.m >= 4:
                        #     break
            # print(self.labels)
        return self.all_list

    def mms(self, all_datas):
        new_datas = []
        for one_datas in all_datas:
            new_datas.append(one_datas[:-1])
        mms = MinMaxScaler(feature_range=(0, 1))
        new_datas = mms.fit_transform(new_datas)
        new_datas = new_datas.tolist()
        for i, datas in enumerate(new_datas):
            datas.append(all_datas[i][-1])
        return new_datas

    def pca(self, all_datas):
        all_d = []
        pca = PCA(n_components=0.99)
        for datas in all_datas:
            feat_datas = datas[:-1]
            all_d.append(feat_datas)
        data = pca.fit_transform(all_d)
        return data.tolist()

    def vt(self, all_datas):
        '''
        过滤式(删除低方差的特征)
        :return: None
        '''
        vt = VarianceThreshold(threshold=1.0)  # hreshold=0.0 过滤方差为0的特征
        data = vt.fit_transform(all_datas)
        return data.tolist()

    def array_txt(self, datas):
        with open("txt/GSE98455.txt", "w") as file:
            for data in datas:
                data_list = list(map(str, data))
                # print(data_list, type(data_list))
                data_str = ",".join(data_list)
                # print(data_str, type(data_str))
                file.write(data_str + "\n")

    def read_arff_label(self, file_name):
        i = 0
        with open("arff/" + file_name + ".arff", "r") as arff_file:
            for line in arff_file:
                i += 1
                # print(line)
                line_str = line.strip("\n")
                print(line_str)
                with open("txt/labels.txt", "a+") as f:
                    f.write(line_str+"\n")
                    if i == 30365:
                        break


# if __name__ == '__main__':
#     #     rd = RD()
#     #     # data = rd.read_csv('spambase')
#     #     data = rd.read_arff('Colon')
#     #     print(len(data))
#     #     for i in range(len(data)):
#     #         print("%d" % i, data[i])
#     #     all_list = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
#     #     all_list = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
#     rd = RD()
#     datas1 = rd.read_arff("Leukemia")
#     datas = rd.vt(datas1)
#     # print(datas)
#     print("原始特征数：", len(datas1[0]), len(datas1))
#     print("原始特征数：", len(datas1[1]), len(datas1))
#     print("处理后的特征数：", len(datas[0]), len(datas))
#     print("处理后的特征数：", len(datas[1]), len(datas))
#
#     # datas = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
#     # datas2 = map(str, datas)
#
#     # print(data.shape)
#
#     # with open("txt/label.txt", "w") as file:
#     #     for data in datas:
#     #         data_list = list(map(str, data))
#     #         # print(data_list, type(data_list))
#     #         data_str = ",".join(data_list)
#     #         # print(data_str, type(data_str))
#     #         file.write(data_str + "\n")


if __name__ == '__main__':
    all_list = [[2, 8, 4, 0], [6, 3, 0, 1], [5, 4, 9, 0]]
    rd = RD()
    datas = rd.mms(all_list)
    print(datas)








