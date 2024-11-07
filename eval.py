import torch
import argparse
import dashscope
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import HfApi


from model_eval import cot_eval, step_eval, get_image_label
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModel, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor


parser = argparse.ArgumentParser(description='Dataset evaluation on MLLMs.')
parser.add_argument('--dashscope', type=str, help='Dashscope api key.')
parser.add_argument('--openai', type=str, help='OpenAI api key.')
parser.add_argument('--hf_token', type=str, help='Huggingface token.')
parser.add_argument("--eval_model", type=str, help="Evaluation model name.")
parser.add_argument("--eval_dataset", type=str, help="Evaluation dataset name, reasoning or description.")
parser.add_argument("--eval_task", type=str, help="Evaluation task name.")
parser.add_argument("--output_path", type=str, help="Evaluation task name.")
parser.add_argument("--setting", type=str, help="zero-shot, few-shot or textual few-shot.")
parser.add_argument("--num_samples", type=int, help="zero-shot or few-shot.", default=None)
args = parser.parse_args()


client = OpenAI(api_key = args.openai)
client.api_key = args.openai
dashscope.api_key = args.dashscope
hf_api = HfApi(token=args.hf_token)


class EvalArgs:
    def __init__(self, eval_model, eval_task, eval_dataset, output_path, setting, num_samples, image_load, model=None, processor=None):
        self.eval_model = eval_model
        self.eval_task = eval_task
        self.eval_dataset = eval_dataset
        self.output_path = output_path
        self.setting = setting
        self.num_samples = num_samples
        self.image_load = image_load
        self.model = model
        self.processor = processor


def eval(eval_args):
    if eval_args.eval_task == "cot":
        cot_eval(eval_args)
    else:
        step_eval(eval_args)


def main():
    image_load = get_image_label(args.eval_task)

    if args.eval_model == "llava-next-7b":
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)       
        model.to("cuda:0")       
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load, model, processor)
    elif args.eval_model == "llava-next-34b":
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to("cuda:0")
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load, model, processor)
    elif args.eval_model == "gemini":
        model = genai.GenerativeModel('gemini-1.5-pro')
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load, model)
    elif args.eval_model == "minicpm":
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load, model, tokenizer)
    elif args.eval_model == "llama":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load, model, processor)
    else:
        eval_args = EvalArgs(args.eval_model, args.eval_task, args.eval_dataset, args.output_path, args.setting, args.num_samples, image_load)

    eval(eval_args)


if __name__ == "__main__":
    main()

