import torch
from transformers import BertModel

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#加载预训练模型
pretrained = BertModel.from_pretrained(r"/Users/chaors/Development/AIProjects/AIentimentAnalysis/model/bert-base-chinese/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea").to(DEVICE)
# print(pretrained)

# 定义下游任务（增量模型）“冻结基座+训练顶层"的模式
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设计全连接网络，对输入数据进行线性变换
        # 实现二分类任务 in_features是BERT模型的输出参数，我们做的是文本情感二分类，所以我们的模型输出是2
        # 768 -> 2的具体含义：nn.Linear的设计遵循一个简单而强大的原则：无论输入张量前面有多少个维度，它都只对最后一个维度进行变换，并将前面的所有维度原封不动地传递到输出
        self.fc = torch.nn.Linear(768,2)
    # 使用模型处理数据（执行前向计算）
    def forward(self,input_ids,attention_mask,token_type_ids):
        # 冻结Bert模型的参数，让其不参与梯度更新。
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])
        return out