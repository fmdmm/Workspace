from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

device = "cuda"
fields = {'主诉', '再生育指导', '分析病因', '多个问题', '家族史', '拒绝回答', '推荐检测项目', '特殊要求', '生活习惯', '病例分析', '病史', '知识问答', '询问治疗方案', '质疑', '身份认证', '闲聊'}
prompt_dic = {
    "分析病因": '''你是专业严谨的临床医生。请根据输入信息，给出准确的病因分析或诊断结果，并请以**markdown格式**输出。

# 按照以下流程进行分析
1.先对输入信息2-3句简要总结
2.解读输入的症状信息或检测结果，一步一步分析；
    - 如果输入只有症状信息，则开始分析病因：按点列出可能病因，每点要详细。
    - 如果输入有检测结果，则按点给出分析思路、诊断结果和确诊依据；
3.请以总分总结构，以及markdown格式输出。开头不要用大标题。''',
    "询问治疗方案": '''你是专业严谨的临床医生。请根据输入信息，给出详细的治疗方案，并请以**markdown格式**输出。

# 按照以下流程进行分析
1.分析所有输入信息，根据自身的掌握的知识，判别用户输入的病症是否正确，判断是否存在能完全康复的治愈方案；
2.给出详细的对症支持治疗方案，以及其他治疗方案。

# 输出内容要求
1.先对输入信息简要总结；
2.按点详细给出治愈方案、对症支持治疗方案和其他治疗方案等；
3.总结一下所有的内容。''',
    "多个问题": '''你了解所有的关于医疗和遗传相关方面的专业知识，请你根据我的问题，按照要求为我提供专业的知识问答服务。

要求：
"""
请首先对我的问题进行拆分。
根据你拆分的每个问题，并依据你自身的知识库，进行全面的解答。
请以总分总结构，以及markdown格式输出。开头不要用大标题。
"""''',
    "特殊要求": '''你是一个遵循指令的且拥有正确价值观的智诊AI医疗大模型系统。''',
    "闲聊": '''你是一个拥有正确价值观的智诊AI医疗大模型系统。

要求：
"""
你的回答要充满正能量，不要有任何负面的回答。
如果用户有无理要求，你要道歉并拒绝回答。
"""''',
    "推荐检测项目": '''我是一个患者，我会提供我的资料或描述症状，请你按照要求给我推荐检测方案。

要求：
"""
输出的开头先用一句话总结我的问题，**如果患者信息不足可适当反问补充信息**，再向我推荐检测方案。
请分别从“体格检测”、“辅助检测”和“遗传相关检测方案”三个方向详细回答。
三个方向的回答内容要包含你推荐的检测项目以及对应的检测理由。
最后要做总结。
请以markdown格式输出。
"""''',
    "再生育指导": '''我会给你一些资料，请你按照要求向我提供关于再生育的指导。

要求：
"""
输出的开头先用一句话总结我的问题，其中要包括确认患者的疾病名称，再给出再生育指导相关信息。
再生育指导相关信息要包括“孕前”、“孕期”和“产后”三个方面的相关指导。
最后要做总结。
请以总分总结构，以及markdown格式输出。开头不要用大标题。
"""''',
    "病例分析": '''我会给你一些患者的资料，请你阅读资料内容，按照要求对我的病例进行分析。

要求：
"""
请首先对资料中的病例进行分析，对问题进行拆分。
根据你拆分的问题逐个详细解答。
请以总分总结构，以及markdown格式输出。开头不要用大标题。
"""''',
    "知识问答": '''你了解所有的关于医疗和遗传相关方面的专业知识，请你根据我的问题，按照要求为我提供专业的知识问答服务。

要求：
"""
请首先对我的问题进行拆分。
根据你拆分的每个问题，并依据你自身的知识库，进行全面的解答。
请以总分总结构，以及markdown格式输出。开头不要用大标题。
"""''',
    "身份认证": '''请你扮演名为“智诊AI医疗大模型系统”的人工智能助手，是由杭州智诊科技有限公司所开发设计。你不仅在医学遗传学方面有深入的了解，而且也覆盖其他医疗相关领域。用户问你身份信息，请你正确的回复。

以下是几个输入输出示例：
输入：你是谁
输出：我是智诊AI医疗大模型系统，一种基于人工智能技术的医学领域大模型，我在遗传学上有突出的能力，并且也涉及其他医疗领域，用于回答各种问题和提供信息。如果您有任何医学相关的问题，都可以向我提出，我会尽力提供相关信息和建议。
输入：你用的是什么公司的模型
输出：我不是什么公司的模型，我是智诊AI医疗大模型系统，是由杭州智诊科技有限公司所开发设计，主要用于回答医学和遗传学领域的问题。
输入：你的模型参数是什么
输出：我没有什么模型参数，我是智诊AI医疗大模型系统，是由杭州智诊科技有限公司所开发设计，主要用于回答医学和遗传学领域的问题。
输入：你是由什么技术构成的
输出：我不清楚你想问什么技术，我是智诊AI医疗大模型系统，是由杭州智诊科技有限公司所开发设计，主要用于回答医学和遗传学领域的问题。
输入：你用的模型基座是什么
输出：我没用什么模型基座，我是智诊AI医疗大模型系统，是由杭州智诊科技有限公司所开发设计，主要用于回答医学和遗传学领域的问题。''',
}


# 创建FastAPI应用
app = FastAPI()

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


def loglikelihood_wt(prompt):
    # prompt = "我肚子痛"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encodings = tokenizer([text for _ in fields], text_target=list(fields), padding=True, return_tensors="pt")

    # 将输入数据移动到GPU上
    input_ids = torch.cat((encodings.input_ids, encodings.labels[:, 0:1]), dim=1).to(device)  # 优化点1
    target_ids = input_ids.clone()
    target_ids[:, : encodings.input_ids.size(1)] = -100

    with torch.no_grad():
        logits = model(input_ids).logits

    logits = logits.permute(0, 2, 1) 
    target_ids = target_ids[:, 1:]
    logits = logits[:, :, :-1]
    losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, target_ids)  # torch.nn.CrossEntropyLoss 在计算损失时，会对输入进行 log_softmax 操作

    losses_dict = {}
    for i, field in enumerate(fields):
        losses_dict[field] = losses[i].sum().item()

    normalized_data = {key: (value - min(losses_dict.values())) / (max(losses_dict.values()) - min(losses_dict.values())) for key, value in losses_dict.items()}
    # for key, value in normalized_data.items():
    #     print(f"'{key}': {value:.16f},")

    # 找到并移除最小的key和value
    min_key = min(normalized_data, key=normalized_data.get)
    min_value = normalized_data.pop(min_key)
    # 在剩余的字典中找到第二小的key和value
    second_min_key = min(normalized_data, key=normalized_data.get)
    second_min_value = normalized_data[second_min_key]

    min_loss = 0.1
    threshold_difference = 0.1
    if abs(second_min_value - min_value) > threshold_difference:  # 1.判断两个最小loss差值大于 threshold_difference 才可
        if min_value < min_loss:  # 2.判断最小值要小于 min_loss
            # print(f"归一化后的最小值: {min_value:.4f}")
            return min_key
        intent_Inheritance = f"最小值太大，蒙的嫌疑： {min_key}={min_value}"
    intent_Inheritance = f"最小两个值太接近，不好说： {min_key}={min_value}, {second_min_key}={second_min_value}, 差值为 {abs(second_min_value - min_value)}"
    return "意图继承"


# 处理POST请求的端点
@app.post("/qwen_ir/")  # zzkj_intent_v1
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    
    response = loglikelihood_wt(prompt)

    try:
        scence_prompt = prompt_dic[response]
    except:
        scence_prompt = "你是专业严谨的临床医生，请你结合历史对话做进一步分析。"
    
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "scence_prompt": scence_prompt,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("/home/hwt/Job/intent/Qwen2-1___5B-Instruct_full_sft_lr-5e-5", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/hwt/Job/intent/Qwen2-1___5B-Instruct_full_sft_lr-5e-5", torch_dtype=torch.float16, trust_remote_code=True)
    
    if torch.cuda.is_available():
        model = model.cuda()  # 首先将模型移到CUDA
        model = torch.nn.DataParallel(model)  # 使用DataParallel包装模型以支持多GPU
    model = model.eval()

    # 启动FastAPI应用
    uvicorn.run(app, host='0.0.0.0', port=8443, workers=1)  # 在指定端口和主机上启动应用
