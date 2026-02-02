from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset

class CHDataSet(Dataset):
    #初始化数据集
    def __init__(self,split):
        #从磁盘加载数据
        self.dataset = load_dataset(r"../data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            print("数据名错误！")

    #返回数据集长度
    def __len__(self):
        return len(self.dataset)

    #对每条数据单独做处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text,label

if __name__ == '__main__':
    dataset = CHDataSet("train")
    for data in dataset:
        print(data)