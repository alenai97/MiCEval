import base64
import json
import re
import random
from PIL import Image


def load_json(file_path):
    with open(file_path, "r") as file:
        demo = json.load(file)
    return demo


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_base_data(item):
    idx = item["idx"]
    image_path = item["image"]
    base64_image = encode_image(image_path)
    question = item["question"]
    cot = item["cot_answer"]
    rationale = "".join(step["step"] for step in item["steps"].values())
    return {
        "idx": idx,
        "image_path": image_path,
        "base64_image": base64_image,
        "question": question,
        "rationale": rationale,
        "cot": cot,
    }


def get_step_label(step):
    step_type = step["step_type"]
    if step_type == "description":
        relevance = step["description_relevance_label"]
        correct = step["description_correctness_label"]
        error_type = step["description_error_label"]
        return relevance, correct, error_type
    elif step_type == "reasoning":
        relevance = step["logic_relevance_label"]
        correct = step["logic_correctness_label"]
        informativeness = step["informativeness_label"]
        return relevance, correct, informativeness
    elif step_type == "both":
        description_correct = step["description_correctness_label"]
        error_type = step["description_error_label"]
        d_relevance = step["description_relevance_label"]
        r_relevance = step["logic_relevance_label"]
        logic_correct = step["logic_correctness_label"]
        informativeness = step["informativeness_label"]
        return step_type, d_relevance, r_relevance, description_correct, error_type, logic_correct, informativeness
    else:
        raise ValueError(f"Step type {step_type} not found.")


def get_cot_label(item):
    fully_description_correct = item["fully_description_correct"]
    fully_logic_correct = item["fully_logic_correct"]
    fully_informative = item["fully_informative"]
    fully_relevant = item["fully_relevant"]
    return fully_description_correct, fully_logic_correct, fully_informative, fully_relevant


def get_prompt(eval_args, base_data):  
    question = base_data["question"]
    rationale = base_data["rationale"]
    if eval_args.eval_task != "cot":
        step = base_data["step"]
        previous = base_data["previous"]

    if eval_args.eval_task == "step_type":       
        system_prompt = "You are an helpful assistant skilled at identifying the type of the step in a chain-of-thought rationale.\n" + \
                "We define the following step types:\nDescription: A step is entailed from the image.\nReasoning: A step is entailed from previous steps.\nBoth: A step is entailed from both the image and previous steps.\n" + \
                "Please analyze the chain-of-thought and categorize the step according to these definitions.\n" + \
                "Only return the most likely label of step type without any explaination.\n"
        text_prompt =  f"Question:{question}\n" + \
                f"Rationale:{rationale}\n" + \
                f"Step:{step}\n" + \
                "Step_type:"   
    elif eval_args.eval_task == "description_correct":
        system_prompt = "You are a helpful assistant capable of judging whether the step correctly describes the image.\n" + \
                "We define the following correct labels:\nFully correct: A step without any incorrect information.\nPartially correct: A step contains some incorrect information, but there is still some correct information as well.\nUnsupported: All information is incorrect.\n" + \
                "In this task, you only need to evaluate the information related to the image description in the step; the reasoning information in the step should not be considered.\n" + \
                "Only return the most likely label correctness without any explaination.\n"
        text_prompt = f"Step:{step}\n" + \
                "Output:"    
    elif eval_args.eval_task == "description_relevant":        
        system_prompt = "You are a helpful assistant skilled at evaluating whether step is relevant to the image and effectively contributes to answering the question.\n" + \
                "We define the following labels:\nImage relevant: A step is relevant to the image.\nLogic relevant: A step is relevant to answering the question.\nBoth: A step is relevant to both the image and answering the question.\nNone: A step is irrelevant to both the image and answering the question.\n" + \
                "Please analyze each step and determine its relevance to both the image and the question being addressed." + \
                "Only return the label of relevance without any explaination.\n"
        text_prompt = f"Question:{question}\n" + \
                f"Rationale:{rationale}\n" + \
                f"Step:{step}\n" + \
                "Output:"
    elif eval_args.eval_task == "description_error_type":
        system_prompt = "You are a helpful assistant capable of categorizing errors in image descriptions.\n" + \
                "We define the following types of errors:\nEntity false: Some entities mentioned in this step are not existing in the image.\nAttribute false: The attributes of an entity are incorrectly described, such as state, color, type, material, shape, size, count, texture.\nSpatial Relationship false: The spatial relationship between two objects is incorrectly described.\nNon-Spatial Relationship false: The active, passive or action relationship between two objects is incorrectly described.\n" + \
                "You can select multiple labels, which means that a step may have multiple errors at the same time.\n" + \
                "Please return the labels of error type without any explaination.\n"
        text_prompt = f"Step:{step}\n" + \
                "Output:"     
    elif eval_args.eval_task == "logic_relevant":
        system_prompt = "You are a helpful assistant skilled at evaluating whether step is relevant to answering the question. Please analyze the step and determine its relevance to the answer.\n" + \
                "We define the following criteria:\nRelevant: A step is relevant to answering the question.\nIrrelevant: A step is not relevant to answering the question.\n" + \
                "Only return the most likely label of relevance without any explaination.\n"
        text_prompt = f"Question:{question}\n" + \
                f"Rationale:{rationale}\n" + \
                f"Step:{step}\n" + \
                "Is this step relevant to answering the question? Output:"    
    elif eval_args.eval_task == "logic_correct":
        system_prompt = "You are a helpful assistant capable of evaluating the logical correctness of each step.\n" + \
                "We define the following criteria:\nCorrect: A step can be logically inferred from the previous steps and without any logical errors or conflicts between its internal clauses.\nIncorrect: A step can not be logically inferred from the previous steps or contains logical errors or conflicts between its internal clauses.\n" + \
                "In this task, you only need to evaluate the information related to the previous steps; the descriptive information in the step should not be considered.\n" + \
                "Please return the most likely label of correctness without any explaination.\n"
        text_prompt = f"Premise:{previous}\n" + \
                f"Hypothesis:{step}\n" + \
                "Output:"
    elif eval_args.eval_task == "logic_error_type":
        system_prompt = "You are a helpful assistant capable of categorizing errors in logic steps.\n" + \
                "We define the following types of errors:\nInter-step Incorrect: A step cannot be logically inferred from the previous steps.\nIntra-step Incorrect: A step contains logical errors or conflicts between its internal clauses.\nBoth: A step cannot be logically inferred from the previous steps and contains logical errors or conflicts between its internal clauses.\n" + \
                "Please return the labels of error type without any explaination.\n"
        text_prompt = f"Premise:{previous}\n" + \
                f"Hypothesis:{step}\n" + \
                "Output:"
    elif eval_args.eval_task == "informativeness":
        system_prompt = "You are a helpful assistant capable of detecting whether each step contains repetition or redundancy compared to previous steps.\n" + \
                "Yes: A step is repetitive or redundant; Otherwise, classify it as No.\n" + \
                "Please return the most likely label of informativeness without any explaination.\n"
        text_prompt = f"Previous:{previous}\n" + \
                f"Step:{step}\n" + \
                "Is this step redundant or repeated? Output:"    
    elif eval_args.eval_task == "cot":
        system_prompt = "You are a helpful assistant capable of evaluating the rationale.\n" + \
                "We define the following criteria:\nYes: The rationale is fully correct, free of any descriptive or logical errors, fully relevant to both the image and the answer, and contains no repetition or redundancy.\nNo: The rationale is either irrelevant to the image or the answer, contains descriptive or logical inaccuracies, or includes repetition or redundancy.\n" + \
                "Please return 'Yes' or 'No' without any explaination.\n"
        text_prompt = f"Question:{question}\n" + \
                f"Rationale:{rationale}\n" + \
                "Is this a good rationale or not? Output:"
    elif eval_args.eval_task == "description_robust":
        system_prompt = "You are a helpful assistant capable of categorizing errors in image descriptions.\n" + \
                "We define the following labels:\nEntity false: Some entities mentioned in this step are not existing in the image.\nAttribute false: The attributes of an entity are incorrectly described, such as state, color, type, material, shape, size, count, texture.\nSpatial Relationship false: Tthe spatial relationship between two objects is incorrectly described.\nNon-Spatial Relationship false: The active, passive or action relationship between two objects is incorrectly described.\nFully correct: A step without any incorrect information.\n" + \
                "You can select multiple labels, which means that a step may have multiple errors at the same time.\n" + \
                "Please return the labels of error type without any explaination.\n"
        text_prompt = f"Step:{step}\n" + \
                "Output:"
    elif eval_args.eval_task == "logic_robust":
        system_prompt = "You are a helpful assistant capable of evaluating the logical correctness of each step.\n" + \
                "We define the following label:\nCorrect: A step can be logically inferred from the previous steps and without any logical errors or conflicts between its internal clauses.\nInter-step incorrect: A step cannot be logically inferred from the previous steps.\nIntra-step incorrect: A step contains logical errors or conflicts between its internal clauses.\nBoth: A step can not be logically inferred from the previous steps and contains logical errors or conflicts between its internal clauses.\nIncorrect: A step can not be logically inferred from the previous steps or contains logical errors or conflicts between its internal clauses.\n" + \
                "In this task, you only need to evaluate the information related to the previous steps; the descriptive information in the step should not be considered.\n" + \
                "Please return the most likely label of correctness without any explaination.\n"
        text_prompt = f"Premise: {previous}\n" + \
                f"Hypothesis:{step}\n" + \
                "Output:" 
    else:
        raise ValueError(f"Task {eval_args.eval_task} not found.")
    prompts = [system_prompt, text_prompt]
    return tuple(prompts)


def process_llava(original_answer):
    cleaned_answer = re.sub(r'\[INST\].*?\[/INST\]', '', original_answer, flags=re.DOTALL).strip()
    return cleaned_answer


def process_llava_34b(original_answer):
    cleaned_answer = re.sub(r'<\|im_start\|>.*?<\|im_start\|>\s*assistant[\r\n]', '', original_answer, flags=re.DOTALL).strip()
    return cleaned_answer


def process_qwen(original_answer):
    if original_answer["text"]:
        cleaned_answer = original_answer["text"]
    else:
        cleaned_answer = original_answer
    return cleaned_answer

def get_image_label(task):
    if task in ["step_type", "description_correct", "description_error_type", "description_relevant", "cot", "description_robust"]:
        return True
    else:
        return False

def get_task_prompt(eval_args):
    demo = load_json("dataset/demo/demo.json")
    num_samples = eval_args.num_samples
    prompt = demo[eval_args.eval_task]  
    if eval_args.eval_task in ["step_type", "description_correct", "logic_error_type"]:        
        a = random.sample(prompt["0"], num_samples)
        b = random.sample(prompt["1"], num_samples)
        c = random.sample(prompt["2"], num_samples)
        task_prompt = a + b + c
    elif eval_args.eval_task in ["logic_relevant", "logic_correct", "informativeness", "cot"]:
        a = random.sample(prompt["0"], num_samples)
        b = random.sample(prompt["1"], num_samples)
        task_prompt = a + b
    elif eval_args.eval_task in ["description_error_type", "description_relevant", "logic_robust"]:
        a = random.sample(prompt["0"], num_samples)
        b = random.sample(prompt["1"], num_samples)
        c = random.sample(prompt["2"], num_samples)
        d = random.sample(prompt["3"], num_samples)        
        task_prompt = a + b + c + d
    elif eval_args.eval_task in ["description_robust"]:
        a = random.sample(prompt["0"], num_samples)
        b = random.sample(prompt["1"], num_samples)
        c = random.sample(prompt["2"], num_samples)
        d = random.sample(prompt["3"], num_samples)
        e = random.sample(prompt["4"], num_samples)        
        task_prompt = a + b + c + d + e
    else:
        raise ValueError("eval_task error.")
    return task_prompt


def get_conversation(eval_args, system_prompt, text_prompt):
    if eval_args.setting != "zero-shot":
        prompt_list = get_task_prompt(eval_args)
    if eval_args.eval_model in ["llava-next-7b", "llava-next-34b"]:
        if eval_args.eval_model == "llava-next-7b":
            conversation = [{"role": "user","content": [{"type": "text", "text": f"{system_prompt}"},],}]
        else:
            conversation = [{"role": "system","content": [{"type": "text", "text": f"{system_prompt}"},],}]
        image_list = []
        if eval_args.setting != "zero-shot":
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = Image.open(prompt["image"])
                    conversation.append({"role": "user","content": [{"type": "image"},{"type": "text", "text": f"{user}"},],})
                    conversation.append({"role": "assistant","content": [{"type": "text", "text": f"{assistant}"},],})
                    image_list.append(image)
                else:
                    conversation.append({"role": "user","content": [{"type": "text", "text": f"{user}"},],})
                    conversation.append({"role": "assistant","content": [{"type": "text", "text": f"{assistant}"},],})
        return conversation, image_list
    elif eval_args.eval_model == "minicpm":
        msgs = []
        if eval_args.setting != "zero-shot":
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = Image.open(prompt["image"])
                    msgs.append({'role': 'user', 'content': [image, user]})
                    msgs.append({'role': 'assistant', 'content': [assistant]})
                else:
                    msgs.append({'role': 'user', 'content': [user]})
                    msgs.append({'role': 'assistant', 'content': [assistant]})
        return msgs
    elif eval_args.eval_model == "gpt-4o":
        conversation = [{"role": "system","content": f"{system_prompt}"}]
        if eval_args.setting != "zero-shot":
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = encode_image(prompt["image"])
                    conversation.append({"role": "user","content": [{"type": "text", "text": f"{user}"},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image}","detail": "low"},},],})
                    conversation.append({"role": "assistant","content": [{"type": "text", "text": f"{assistant}"},],})
                else:
                    conversation.append({"role": "user","content": [{"type": "text", "text": f"{user}"},],})
                    conversation.append({"role": "assistant","content": [{"type": "text", "text": f"{assistant}"},],})
        return conversation
    elif eval_args.eval_model == "qwen-vl-max":
        messages = [{"role": "system","content": [{"text":f"{system_prompt}"}]}]   
        if eval_args.setting != "zero-shot":
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = prompt["image"]
                    messages.append({"role": "user","content":[{"image": image},{"text": f"{text_prompt}"} ]})
                    messages.append({"role": "assistant","content": [{"text":f"{assistant}"}]})
                else:
                    messages.append({"role": "user","content":[{"text": f"{user}"}]})
                    messages.append({"role": "assistant","content": [{"text":f"{assistant}"}]})
        return messages
    elif eval_args.eval_model == "gemini":
        contents = [system_prompt]
        if eval_args.setting != "zero-shot":
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = encode_image(prompt["image"])
                    contents.append(image)
                    contents.append(user)
                    contents.append(assistant)
                else:
                    contents.append(user)
                    contents.append(assistant)
        return contents
    elif eval_args.eval_model == "llama":
        messages = [{"role": "user", "content": [{"type": "text", "text": f"{system_prompt}"}]}]
        image_list = []
        if eval_args.setting != "zero-shot":       
            for prompt in prompt_list:
                user = prompt["user"]
                assistant = prompt["assistant"]
                if eval_args.image_load and eval_args.setting == "few-shot":
                    image = Image.open(prompt["image"])
                    messages.append({"role": "user", "content": [{"type": "image"},{"type": "text", "text": f"{text_prompt}"}]})
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{assistant}"}]})
                    image_list.append(image)
                else:
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"{text_prompt}"}]})
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{assistant}"}]})
        return messages, image_list
