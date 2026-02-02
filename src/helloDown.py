from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.configuration_auto import model_type_to_module_name

# 将模型和分词器下载到本地使用，指定保存路径
# model_name = "uer/gpt2-chinese-cluecorpussmall"
# cache_dir = "./model/uer/gpt2-chinese-cluecorpussmall"
# model_name = "google-bert/bert-base-chinese"
# cache_dir = "./model/google-bert/bert-base-chinese"
model_name = "bert-base-chinese"
cache_dir = "./model/bert-base-chinese"

# 下载模块
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# 下载分词工具
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型已经下载到：{cache_dir}")

# .no_exist  配置文件
# 核心模型文件 snapshots
# safeetensors huggingface模型文件
# pytorch_model python模型文件   一般config在哪就用哪个
# vocab_size 模型字典大小，模型能识别的字符数量
# n_positions 1024
