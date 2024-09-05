import torch 
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist 
import transformers 
from accelerate import Accelerator 
import lm_eval 
from datasets import load_dataset 
# from transformers import AutoTokenizer, LlamaForCausalLM 
from transformers import AutoTokenizer 
from llama12addingtree import get_llama_griffin, get_llama_griffin2, LlamaForCausalLM 
from llama12 import get_llama_griffin_no_tree, get_llama_griffin2_no_tree, LlamaForCausalLMNoTree 
from llama12addingtree import MultiTokenEOSCriteria 
# from llama12 import LlamaForCausalLM 
import numpy as np 
from datasets import concatenate_datasets 
from datasets import Dataset 
from torch.utils.data import DataLoader 
from typing import List, Literal, Optional, Tuple, Union 
import argparse 
from tqdm import tqdm 
from termcolor import colored 
from tabulate import tabulate 
import copy 
import os 
import random 

def set_seed(seed):
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        
    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# Usage
set_seed(42)  # You can use any integer value as the seed 

### Parsing the arguments ### 
parser = argparse.ArgumentParser(description = "CommonSense Reasoning with generation and chain-of-thoughts") 
parser.add_argument("--tasks", type = str) 
parser.add_argument("--model", type = str) 
parser.add_argument("--device", type = str, default = None) 
parser.add_argument("--limit", type = int, default = None) 
# parser.add_argument("--griffin", type = bool, default = False) 
parser.add_argument("--griffin", action = "store_true") 
# parser.add_argument("--cats", type = bool, default = False) 
parser.add_argument("--cats", action = "store_true") 
# parser.add_argument("--check", type = bool, default = False) 
parser.add_argument("--check", action = "store_true") 
parser.add_argument("--kernel_size", type = int, default = None) 
parser.add_argument("--spr", type = float, default = 0.5) 
parser.add_argument("--thr", type = float, default = 0.1) 
parser.add_argument("--widthtree", type = int, default = 8) 
parser.add_argument("--patternstrict", action = "store_true") 
parser.add_argument("--shotfive", action = "store_true") 
parser.add_argument("--shottwo", action = "store_true") 
parser.add_argument("--filteractiveenabled", action = "store_true") 

os.environ['NCCL_TIMEOUT'] = '1800'  # For 2 hours 
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
print("NCCL_TIMEOUT {}".format(os.environ['NCCL_TIMEOUT'])) 

from accelerate.utils import InitProcessGroupKwargs 
from datetime import timedelta 
# from accelerate.utils import DistributedDataParallelKwargs 

kwargs = InitProcessGroupKwargs(timeout = timedelta(minutes = 60)) 
accelerator = Accelerator(kwargs_handlers=[kwargs]) 

# Check if we are in a distributed setup
is_distributed = accelerator.distributed_type != "NO" 
print("is_distributed {}".format(is_distributed)) 

args = parser.parse_args() 
tasks = args.tasks.split(",") 
print(args) 

if args.device is None: 
    # args.device = "cuda" if torch.cuda.is_available() else "cpu" 
    if is_distributed: 
        args.device = "cuda:{}".format(accelerator.process_index) if torch.cuda.is_available() else "cpu" 
    else: 
        args.device = "cuda" if torch.cuda.is_available() else "cpu" 

### Loading the tokenizer and the model ### 
tokenizer = AutoTokenizer.from_pretrained(args.model) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

if args.check: 
    model = LlamaForCausalLM.from_pretrained(args.model, device_map = args.device, torch_dtype = torch.bfloat16) 
else: 
    model = LlamaForCausalLMNoTree.from_pretrained(args.model, device_map = args.device, torch_dtype = torch.bfloat16) 

if args.griffin: 
    schedule_k = [args.spr for _ in range(model.config.num_hidden_layers)] 
if args.cats: 
    schedule_k = [(1 - args.spr) for _ in range(model.config.num_hidden_layers)] 

model.config.mode = "gen" 
model.config.selection_method = "topk" 
model.config.check = args.check 
model.config.griffin = args.griffin 
model.config.kernel_size = args.kernel_size 
model.config.thr = args.thr 
model.config.secondrollback = False 
model.config.treewidth = args.widthtree # here we set the width of the tree 
model.config.filteractiveenabled = args.filteractiveenabled # only used for 8B model 

if args.check: 
    if args.griffin: 
        model = get_llama_griffin2(model, schedule_k) 
    if args.cats: 
        model = get_llama_griffin(model, schedule_k, patternstrict = args.patternstrict) 
        # model = get_llama_griffin(model, schedule_k, patternstrict = args.patternstrict) 
else: 
    if args.griffin: 
        model = get_llama_griffin2_no_tree(model, schedule_k) 
    if args.cats: 
        model = get_llama_griffin_no_tree(model, schedule_k) 
        # model = get_llama_griffin(model, schedule_k, patternstrict = args.patternstrict) 

model.eval() 
if is_distributed: 
    model = accelerator.prepare(model) 

def compensatingdataset(dataset, datasetname): 
    if len(dataset) % accelerator.num_processes == 0 or not is_distributed: 
        return dataset 
    else: 
        lengthdummy = accelerator.num_processes - (len(dataset) % accelerator.num_processes) 
        # datasetdummy = dataset.copy() 
        if datasetname == "csqa": 
            datasetdummy = load_dataset("tau/commonsense_qa", split = "validation[:{}]".format(lengthdummy)) 
        elif datasetname == "strategyqa": 
            datasetdummy = load_dataset("tasksource/bigbench", "strategyqa", split = "validation[:{}]".format(lengthdummy)) 
        elif datasetname == "date": 
            datasetdummy = load_dataset("tasksource/bigbench", "date_understanding", split = "train[:{}]".format(lengthdummy)) 
        elif datasetname == "sports": 
            datasetdummy = load_dataset("tasksource/bigbench", "sports_understanding", split = "train[:{}]".format(lengthdummy)) 
        elif datasetname == "aqua": 
            datasetdummy = load_dataset("deepmind/aqua_rat", split = "test[:{}]".format(lengthdummy)) 
        else: 
            raise ValueError("Unknown dataset {}".format(datasetname)) 
        def addingsignal(example): 
            example["keep"] = "n" 
            return example 
        datasetdummy = datasetdummy.map(addingsignal) 
        dataset = concatenate_datasets([dataset, datasetdummy]) 
        return dataset 

### Loading the datasets ### 
def get_dataset(datasetname, is_distributed = False, requirements = ""): 
    # loading the manually written cot prompt 
    cotprompt: str = None 
    with open("{}_cot_prompts{}.txt".format(datasetname, requirements), "r") as file: 
        cotprompt = file.read() 
        cotprompt = cotprompt.replace("\\n", "") 
        cotprompt = cotprompt.replace("\\", "") 
    if datasetname == "csqa": 
        # loading the actual dataset 
        if args.limit is None: 
            dataset = load_dataset("tau/commonsense_qa", split = "validation") 
        else: 
            dataset = load_dataset("tau/commonsense_qa", split = "validation[:{}]".format(args.limit)) 
        def annotatedataset(example): 
            newentry = {} 
            newentry["keep"] = "y" 
            return newentry 
        dataset = dataset.map(annotatedataset, num_proc = 8) 
        if is_distributed: 
            dataset = compensatingdataset(dataset, datasetname) 
        def encodewithtokenizer(example): 
            options = example["choices"]["text"] 
            inputtext = "Q: {}\nOptions: (a) {} (b) {} (c) {} (d) {} (e) {}\nA:".format(example["question"], options[0], options[1], options[2], options[3], options[4]) 
            outputdi = tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
            return outputdi 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
        
        print("length of dataset: ", len(dataset)) 
    elif datasetname == "strategyqa": 
        if args.limit is None: 
            dataset = load_dataset("tasksource/bigbench", "strategyqa", split = "validation") 
        else: 
            dataset = load_dataset("tasksource/bigbench", "strategyqa", split = "validation[:{}]".format(args.limit)) 
        def annotatedataset(example): 
            newentry = {} 
            newentry["keep"] = "y" 
            return newentry 
        dataset = dataset.map(annotatedataset, num_proc = 8) 
        if is_distributed: 
            dataset = compensatingdataset(dataset, datasetname) 
        def encodewithtokenizer(example): 
            inputtext = "Q: Yes or No: {}".format(example["inputs"][3 :]) 
            outputdi = tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
            return outputdi 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "date": 
        dataset = load_dataset("tasksource/bigbench", "date_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def annotatedataset(example): 
            newentry = {} 
            newentry["keep"] = "y" 
            return newentry 
        dataset = dataset.map(annotatedataset, num_proc = 8) 
        if is_distributed: 
            dataset = compensatingdataset(dataset, datasetname) 
        def encodewithtokenizer(example): 
            inputtext = example["inputs"] 
            outputdi = tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
            return outputdi 
        dataset = dataset.select(range(10, len(dataset))) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "sports": 
        dataset = load_dataset("tasksource/bigbench", "sports_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def annotatedataset(example): 
            newentry = {} 
            newentry["keep"] = "y" 
            return newentry 
        dataset = dataset.map(annotatedataset, num_proc = 8) 
        if is_distributed: 
            dataset = compensatingdataset(dataset, datasetname) 
        def encodewithtokenizer(example): 
            # inputtext = "Q: {}".format(example["inputs"]) 
            inputtext = "Q: {}\nA:".format(example["inputs"]) 
            outputdi = tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
            return outputdi 
        dataset = dataset.select(range(10, len(dataset))) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
        
    elif datasetname == "aqua": 
        dataset = load_dataset("deepmind/aqua_rat", split = "test") 
        # dataset = concatenate_datasets([dataset["validation"], dataset["test"]]) 
        def annotatedataset(example): 
            newentry = {} 
            newentry["keep"] = "y" 
            return newentry 
        dataset = dataset.map(annotatedataset, num_proc = 8) 
        if is_distributed: 
            dataset = compensatingdataset(dataset, datasetname) 
        def encodewithtokenizer(example): 
            options = example["options"] 
            inputtext = "Q: {}\nOptions: (a) {} (b) {} (c) {} (d) {} (e) {}\nA:".format(example["question"], options[0][2 : ], options[1][2 : ], options[2][2 : ], options[3][2 : ], options[4][2 : ]) 
            outputdi = tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
            return outputdi 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    else: 
        raise ValueError("Unknown dataset {}".format(datasetname)) 
    
    if is_distributed: 
        distributedsampler = DistributedSampler(dataset, num_replicas = accelerator.num_processes, rank = accelerator.process_index, drop_last = True, shuffle = True) 
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, sampler = distributedsampler) 
    else: 
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False) 
    return dataloader, cotprompt 

class MaxLengthCriteria(transformers.StoppingCriteria): 
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
    max_length: int = 256, 
) -> transformers.StoppingCriteriaList:
    outputstoppingcriteria = transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    ) 
    
    # max_length_criteria = MaxLengthCriteria(max_length) 
    # outputstoppingcriteria.append(max_length_criteria) 
    
    return outputstoppingcriteria 

def criteriaoutput(datasetname, outputs, inputexample): 
    if datasetname == "csqa": 
        expectedanswer = inputexample["answerKey"][0].lower() 
        generatedtext = tokenizer.decode(outputs) 
        generatedtext = generatedtext.lower() 
        indexpinned = generatedtext.find("so the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        # answer = generatedtext[indexpinned + len("So the answer is ") : indexperiod] 
        answer = generatedtext[indexperiod - 2] 
        # expectedanswer = batch["answerKey"][0].lower() 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    elif datasetname == "strategyqa": 
        expectedanswer = inputexample["multiple_choice_targets"][inputexample["multiple_choice_scores"].index(1)][0].lower() 
        generatedtext = tokenizer.decode(outputs) 
        generatedtext = generatedtext.lower() 
        indexpinned = generatedtext.find("so the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("so the answer is ") : indexperiod] 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    elif datasetname == "date": 
        expectedanswer = inputexample["targets"][0][0] 
        generatedtext = tokenizer.decode(outputs) 
        generatedtext = generatedtext.lower() 
        indexpinned = generatedtext.find("so the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("so the answer is ") : indexperiod] 
        resultoutput = False 
        if answer == expectedanswer: 
            resultoutput = True 
        else: 
            segsanswer = answer.split("/") 
            segsexpectedanswer = expectedanswer.split("/") 
            # print("length of answer {} expected {}".format(len(segsanswer), len(segsexpectedanswer))) 
            if len(segsanswer) != len(segsexpectedanswer): 
                # print("length of answer {} expected {}".format(len(segsanswer), len(segsexpectedanswer))) 
                resultoutput = False 
            else: 
                # print("entering the else") 
                accumulate = True 
                try: 
                    for i in range(3): 
                        if segsexpectedanswer[i][0] == '0': 
                            segsexpectedanswer[i] = segsexpectedanswer[i][1 : ] 
                        if segsanswer[i][0] == '0': 
                            segsanswer[i] = segsanswer[i][1 : ] 
                        accumulate = accumulate and (segsanswer[i] == segsexpectedanswer[i]) 
                        # print("answer {} expected {} accumulate {}".format(segsanswer[i], segsexpectedanswer[i], accumulate)) 
                except: 
                    accumulate = False 
                resultoutput = accumulate 
        if accelerator.is_main_process: 
            if resultoutput: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(resultoutput) 
    elif datasetname == "sports": 
        expectedanswer = inputexample["targets"][0][0] 
        generatedtext = tokenizer.decode(outputs) 
        generatedtext = generatedtext.lower() 
        indexpinned = generatedtext.find("so the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("so the answer is ") : indexperiod] 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    elif datasetname == "aqua": 
        expectedanswer = inputexample["correct"] 
        expectedanswer = expectedanswer[0].lower() 
        generatedtext = tokenizer.decode(outputs) 
        generatedtext = generatedtext.lower() 
        indexpinned = generatedtext.find("so the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        if indexpinned == -1 or indexperiod == -1: 
            answer = "" 
        else: 
            answer = generatedtext[indexpinned + len("so the answer is ") : indexperiod] 
            answer = answer if len(answer) == 1 else answer[1] 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    else: 
        raise ValueError("Unknown dataset {}".format(datasetname)) 

print("tasks {}".format(tasks)) 
countaccum = {} 
for task in tasks: 
    if is_distributed: 
        model.module.updatestatistic() 
    else: 
        model.updatestatistic() 
    # dataloader, cotprompt = get_dataset(task, requirements = "_5shot") 
    if args.shotfive: 
        dataloader, cotprompt = get_dataset(task, is_distributed = is_distributed, requirements = "_5shot") 
    elif args.shottwo: 
        dataloader, cotprompt = get_dataset(task, is_distributed = is_distributed, requirements = "_1shot") 
    else: 
        dataloader, cotprompt = get_dataset(task, is_distributed = is_distributed, requirements = "") 
    promptids = tokenizer(cotprompt, return_tensors = "pt", truncation = True, padding = False)["input_ids"] 
    promptids = torch.tensor(promptids, dtype = torch.long).to(args.device) 
    totalexamples = 0 
    correctanswers = 0 
    
    '''
    # make the kv cache 
    outputs = model(
        input_ids = promptids, 
        use_cache = True, 
        return_dict = True, 
    ) 
    kv_cache = outputs.past_key_values 
    ''' 
    
    for i, batch in enumerate(tqdm(dataloader)): 
        if batch["keep"][0] == "n": 
            print(colored("Skipping the batch", "yellow")) 
            continue 
        # print("answer found {}".format("answerKey" in batch.keys())) 
        # print(batch["answerKey"][0]) 
        # print(len(batch["answerKey"])) 
        # exit(0) 
        input_ids = batch["input_ids"] 
        input_ids = torch.tensor(input_ids, dtype = torch.long) 
        input_ids = input_ids.to(args.device) 
        # if accelerator.is_main_process: 
            # print(tokenizer.decode(input_ids[0])) 
        input_ids = torch.cat([promptids, input_ids], dim = 1) 
        input_ids = input_ids.to(args.device) 
        # stop_criteria = stop_sequences_criteria(tokenizer, "Q:", input_ids.shape[1], input_ids.shape[0]) 
        stop_criteria = stop_sequences_criteria(tokenizer, ["Q:"], input_ids.shape[1], input_ids.shape[0], 256) 
        if is_distributed: 
            outputs = model.module.generate(
                input_ids = input_ids, 
                attention_mask = None, 
                # max_length = input_ids.shape[1] + 20, 
                max_length = input_ids.shape[1] + 256, 
                use_cache = True, 
                stopping_criteria = stop_criteria, 
                pad_token_id = tokenizer.pad_token_id, 
                do_sample = False, 
                # past_key_values = kv_cache, 
            ) 
        else: 
            outputs = model.generate(
                input_ids = input_ids, 
                attention_mask = None, 
                max_length = input_ids.shape[1] + 256, 
                use_cache = True, 
                stopping_criteria = stop_criteria, 
                pad_token_id = tokenizer.pad_token_id, 
                do_sample = False, 
            ) 
        # print(tokenizer.decode(outputs[0])) 
        # if accelerator.is_main_process: 
            # print(tokenizer.decode(outputs[0][input_ids.shape[1] :])) 
        generatedtext = tokenizer.decode(outputs[0][input_ids.shape[1] :]) 
        checkcriteria = criteriaoutput(task, outputs[0][input_ids.shape[1] :], batch) 
        totalexamples += 1 
        correctanswers += checkcriteria 
        # if accelerator.is_main_process: 
            # print("Total examples: {} Correct answers: {}".format(totalexamples, correctanswers)) 
        
        # adding synchronization rounds 
        if is_distributed and i == len(dataloader)//2: 
            dist.barrier() 
    
    # statistics 
    headers = ["Task"] 
    data = [task] 
    if is_distributed: 
        print("index {} start communication".format(accelerator.process_index)) 
        # dist.barrier(timeout=timedelta(minutes=30)) 
        dist.barrier() 
        totalexamples = torch.tensor(totalexamples, device = args.device) 
        correctanswers = torch.tensor(correctanswers, device = args.device) 
        dist.all_reduce(totalexamples, op = dist.ReduceOp.SUM) 
        dist.all_reduce(correctanswers, op = dist.ReduceOp.SUM) 
        totalexamples = totalexamples.item() 
        correctanswers = correctanswers.item() 
        num_sentence = model.module.num_sentence 
        totalgenerationlength = model.module.totalgenerationlength 
        numsentences = torch.tensor([num_sentence, totalgenerationlength], device = args.device) 
        dist.all_reduce(numsentences, op = dist.ReduceOp.SUM) 
        num_sentence = numsentences[0].item() 
        totalgenerationlength = numsentences[1].item() 
        averagegenerationlength = totalgenerationlength / num_sentence 
        headers += ["Num Sentence", "Total Generation Length", "Average Generation Length"] 
        data += [num_sentence, totalgenerationlength, averagegenerationlength] 
        if args.check: 
            total_step = model.module.total_steps 
            num_step = model.module.num_steps 
            totalsteps = torch.tensor([total_step, num_step], device = args.device) 
            dist.all_reduce(totalsteps, op = dist.ReduceOp.SUM) 
            total_step = totalsteps[0].item() 
            num_step = totalsteps[1].item() 
            aal = total_step / num_step 
            headers += ["Total Steps", "Num Steps", "AAL"] 
            data += [total_step, num_step, aal] 
            total_roll_back_length_error = model.module.total_roll_back_length_error 
            errorinstance = model.module.errorinstance 
            totalrollbacklengtherrors = torch.tensor([total_roll_back_length_error, errorinstance], device = args.device) 
            dist.all_reduce(totalrollbacklengtherrors, op = dist.ReduceOp.SUM) 
            total_roll_back_length_error = totalrollbacklengtherrors[0].item() 
            errorinstance = totalrollbacklengtherrors[1].item() 
            averagerollbacklengtherror = total_roll_back_length_error / errorinstance 
            headers += ["Total Roll Back Length Error", "Error Instance", "Average Roll Back Length Error"] 
            data += [total_roll_back_length_error, errorinstance, averagerollbacklengtherror] 
            np.save("{}_rollbacklengthinerror_{}.npy".format(task, accelerator.process_index), np.array(model.module.roll_back_length_in_error)) 
            
            # tree size statistics 
            totaltreesize = model.module.flattentreesize # flattened tree size 
            draftingtreesize = model.module.averagedraftingbatchsize # average drafting batch size 
            totaltreesize = torch.tensor([totaltreesize, draftingtreesize], device = args.device, dtype = torch.float) 
            dist.all_reduce(totaltreesize, op = dist.ReduceOp.SUM) 
            draftingtreesize = totaltreesize[1].item() 
            totaltreesize = totaltreesize[0].item() 
            headers += ["Effective Tree Size", "Drafting Tree Size"] 
            data += [totaltreesize/num_step, draftingtreesize/num_step] 
            
    else: 
        num_sentence = model.num_sentence 
        totalgenerationlength = model.totalgenerationlength 
        averagegenerationlength = totalgenerationlength / num_sentence 
        headers += ["Num Sentence", "Total Generation Length", "Average Generation Length"] 
        data += [num_sentence, totalgenerationlength, averagegenerationlength] 
        if args.check: 
            total_step = model.total_steps 
            num_step = model.num_steps 
            aal = total_step / num_step 
            headers += ["Total Steps", "Num Steps", "AAL"] 
            data += [total_step, num_step, aal] 
            total_roll_back_length_error = model.total_roll_back_length_error 
            errorinstance = model.errorinstance 
            averagerollbacklengtherror = total_roll_back_length_error / errorinstance 
            headers += ["Total Roll Back Length Error", "Error Instance", "Average Roll Back Length Error"] 
            data += [total_roll_back_length_error, errorinstance, averagerollbacklengtherror] 
            np.save("{}_rollbacklengthinerror.npy".format(task), np.array(model.roll_back_length_in_error)) 
    # print("Task\tTotal Steps\tNum Steps\tAAL\tNum Sentence\tTotal Generation Length\tAverage Generation Length\tTotal Roll Back Length Error\tError Instance\tAverage Roll Back Length Error") 
    # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(task, total_step, num_step, aal, num_sentence, totalgenerationlength, averagegenerationlength, total_roll_back_length_error, errorinstance, averagerollbacklengtherror)) 

    # Print table 
    print("Here are the statistics for inference") 
    if accelerator.is_main_process: 
        print(tabulate([data], headers=headers, tablefmt="grid")) 
    # if is_distributed: 
    #     model.module.updatestatistic() 
    # else: 
    #     model.updatestatistic() 
    countaccum[task] = [totalexamples, correctanswers, correctanswers / totalexamples] 

if accelerator.is_main_process: 
    # formatting the output 
    print(args) 
    # print("Task\tTotal\tCorrect\tSolve Rate") 
    headers = ["Task", "Total", "Correct", "Solve Rate"] 
    data = [] 
    for task in tasks: 
        # print("{}\t{}\t{}\t{}".format(task, countaccum[task][0], countaccum[task][1], countaccum[task][2])) 
        data.append([task, countaccum[task][0], countaccum[task][1], countaccum[task][2]]) 
    print(tabulate(data, headers = headers, tablefmt = "grid")) 
