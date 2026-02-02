from datasets import load_dataset,load_from_disk
from pyarrow.compute import index

#在线加载数据,cache_dir为存储目录
# dataset = load_dataset(path="lansinuote/ChnSentiCorp",cache_dir="../data/")
# print(dataset)
# # 转为csv格式
# for name,data in dataset.items():
#     df = data.to_pandas()
#     # 要保证data目录存在
#     df.to_csv(f"../data/{name}.csv",index=False)

# 加载缓存数据
# datasets = load_from_disk(r"/Users/chaors/Development/AIProjects/AIentimentAnalysis/data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a")
# datasets = load_dataset(r"/Users/chaors/Development/AIProjects/AIentimentAnalysis/data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a")
# print(datasets)

# 加载测试数据
# train_data = datasets["train"]
# print(train_data)
# for data in train_data:
    # print(data)

# 扩展：加载CSV格式数据
dataset = load_dataset("csv",data_files=r"../data/validation.csv")
print(dataset)
for data in dataset["train"]:
    print(data)