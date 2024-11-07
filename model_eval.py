import os
import json
from PIL import Image
from http import HTTPStatus

from eval import client, genai, dashscope
from help_function import *



def get_original_data(eval_dataset):
    data = []
    with open(f"dataset/MiCEval_{eval_dataset}.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data



def step_eval(eval_args):
    data = get_original_data(eval_args.eval_dataset)
    results = []
    for item in data:
        base_data = get_base_data(item)
        steps = item["steps"]
        if "previous" not in base_data:
            base_data["previous"] = ""
        for idx in steps:
            cur_step = steps[idx]
            base_data["step"] = cur_step["step"]
            if eval_args.eval_task == "step_type": 
                steps[idx]["model_answer"] = get_model_answer(eval_args, base_data)
            elif eval_args.eval_task == "description_error_type" and cur_step["type_label"] in {"Description", "Both"}:
                if cur_step["description_correctness_label"] != "Fully correct":
                    steps[idx]["model_answer"] = get_model_answer(eval_args, base_data)
            elif eval_args.eval_task == "logic_error_type" and cur_step["type_label"] in {"Reasoning", "Both"}:
                if cur_step["logic_correctness_label"] != "Correct":
                    steps[idx]["model_answer"] = get_model_answer(eval_args, base_data)
            elif eval_args.eval_task in ["description_correct", "description_relevant", "description_robust"] and cur_step["type_label"] in {"Description", "Both"}:
                    steps[idx]["model_answer"] = get_model_answer(eval_args, base_data)
            elif eval_args.eval_task in ["logic_correct", "informativeness", "logic_relevant", "logic_robust"] and cur_step["type_label"] in {"Reasoning", "Both"}:
                    steps[idx]["model_answer"] = get_model_answer(eval_args, base_data)
            base_data["previous"] += cur_step["step"]
        results.append(item)
        

        output_dir = os.path.dirname(eval_args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(eval_args.output_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')



def cot_eval(eval_args):
    data = get_original_data(eval_args.eval_dataset)
    results = []
    for item in data:
        base_data = get_base_data(item)
        model_answer = get_model_answer(eval_args, base_data)
        item["model_answer"] = model_answer       
        results.append(item)

        output_dir = os.path.dirname(eval_args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(eval_args.output_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')



def get_model_answer(eval_args, base_data):
    model_handlers = {
        "gpt-4o": answering_gpt,
        "qwen-vl-max": answering_qwen,
        "llava-next-7b": answering_llava,
        "llava-next-34b": answering_llava,
        "gemini": answering_gemini,
        "minicpm": answering_minicpm,
        "llama": answering_llama
    }

    handler = model_handlers.get(eval_args.eval_model)
    if handler is not None:
        return handler(eval_args, base_data)
    else:
        raise ValueError(f"Model {eval_args.eval_model} not found.")

        

def answering_gpt(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)
    conversation = get_conversation(eval_args, system_prompt, text_prompt)


    if eval_args.image_load:
        base64_image = base_data["base64_image"]
        conversation.append({"role": "user","content": [{"type": "text", "text": f"{text_prompt}"},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}","detail": "auto"},},],})
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=conversation,
            max_tokens=30,
            temperature=0,
        )
        model_answer = response.choices[0].message.content
    else:
        conversation.append({"role": "user","content": [{"type": "text", "text": f"{text_prompt}"},],})
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06", 
            messages=conversation,
            max_tokens=30,
            temperature=0,
        )
    model_answer = response.choices[0].message.content
    return model_answer



def answering_qwen(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)
    messages = get_conversation(eval_args, system_prompt, text_prompt)
    if eval_args.image_load:
        image_path = base_data["image_path"]
        messages.append({"role": "user","content":[{"image": image_path},{"text": f"{text_prompt}"}]})
        response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                        messages=messages,
                                                        max_tokens=30,
                                                        temperature=0)
    else:
        messages.append({"role": "user","content": [{"text": f"{text_prompt}"}]})
        response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                        messages=messages,
                                                        max_tokens=30,
                                                        temperature=0)
    if response.status_code == HTTPStatus.OK:
        original_answer = response["output"]["choices"][0]["message"]["content"][0]
        model_answer = process_qwen(original_answer)
    else:
        print(response.code)
        print(response.message)   
    return model_answer



def answering_llava(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)

    conversation, image_list = get_conversation(eval_args, system_prompt, text_prompt)
        
    if eval_args.image_load:
            conversation.append({"role": "user","content": [{"type": "image"},{"type": "text", "text": f"{text_prompt}"},],})
            image_list.append(Image.open(base_data["image_path"]).convert("RGB"))
    else:
        conversation.append({"role": "user","content": [{"type": "text", "text": f"{text_prompt}"},],})

    prompt = eval_args.processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = eval_args.processor(text=prompt, images=image_list, padding=True, return_tensors="pt").to("cuda:0")
    generate_ids = eval_args.model.generate(**inputs, max_new_tokens=30, temperature=0)
    model_answer = eval_args.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return model_answer



def answering_minicpm(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)
    msgs = get_conversation(eval_args, system_prompt, text_prompt)

    if eval_args.image_load:
        image = Image.open(base_data["image_path"]).convert("RGB")
        msgs.append({'role': 'user', 'content': [image, text_prompt]})
    else:
        msgs = ({'role': 'user', 'content': [text_prompt]})

    model_answer = eval_args.model.chat(
        image=None,
        msgs=msgs,
        tokenizer=eval_args.processor,
        max_tokens=30,
        system_prompt=system_prompt,
        generation_config = {"temperature": 0}
    )
    return model_answer



def answering_gemini(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)
    contents = get_conversation(eval_args, system_prompt, text_prompt)

    if eval_args.image_load:
        image_path = base_data["image_path"]
        image = Image.open(image_path).convert("RGB")
        contents.append(image)
    contents.append(text_prompt)

    response = eval_args.model.generate_content(contents, 
                                                generation_config=genai.types.GenerationConfig(
                                                candidate_count=1,
                                                max_output_tokens=30,
                                                temperature=0))
    model_answer = response.text

    return model_answer



def answering_llama(eval_args, base_data):
    system_prompt, text_prompt = get_prompt(eval_args, base_data)
    messages, image_list = get_conversation(eval_args, system_prompt, text_prompt)
    
    if eval_args.image_load:
        messages.append({"role": "user", "content": [{"type": "image"},{"type": "text", "text": f"{text_prompt}"}]})
        image_list.append(Image.open(base_data["image_path"]).convert("RGB"))
    else:
        image_list = None
        messages.append({"role": "user", "content": [{"type": "text", "text": f"{text_prompt}"}]})
    

    input_text = eval_args.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = eval_args.processor(image_list, input_text, return_tensors="pt").to("cuda:0")

    output = eval_args.model.generate(**inputs, max_new_tokens=30, temperature=0.1)
    model_answer = eval_args.processor.decode(output[0])
    return model_answer
