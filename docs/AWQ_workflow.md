## Tinychat workflow

本文详细解释Tinychat项目的workflow

### 模型参数导入
此部分在tinychat-tutorial/transformer/application/chat.cc中。
![alt text](image.png)

可以看出，LLaMA7B模型由一个Int4LlamaForCausalLM的C++类表示，定义如下：
![alt text](image-1.png)


