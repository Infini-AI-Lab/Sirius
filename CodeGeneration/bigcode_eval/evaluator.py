import inspect
import json
import os
import warnings

from typing import List


from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations 
import torch 
import torch.distributed as dist 
from tabulate import tabulate 

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_generations_path,
        ) 
        print("type of model {}".format(type(self.model))) 
        
        headers = [] 
        data = [] 
        num_sentence = self.model.num_sentence 
        totalgenerationlength = self.model.totalgenerationlength 
        averagegenerationlength = totalgenerationlength / num_sentence 
        numsentences = torch.tensor([num_sentence, totalgenerationlength], device = self.model.device) 
        dist.all_reduce(numsentences, op = dist.ReduceOp.SUM) 
        num_sentence = numsentences[0].item() 
        totalgenerationlength = numsentences[1].item() 
        averagegenerationlength = totalgenerationlength / num_sentence 
        headers += ["Num Sentence", "Total Generation Length", "Average Generation Length"] 
        data += [num_sentence, totalgenerationlength, averagegenerationlength] 
        if self.model.config.check: 
            total_step = self.model.total_steps 
            num_step = self.model.num_steps 
            totalsteps = torch.tensor([total_step, num_step], device = self.model.device) 
            dist.all_reduce(totalsteps, op = dist.ReduceOp.SUM) 
            total_step = totalsteps[0].item() 
            num_step = totalsteps[1].item() 
            aal = total_step / num_step 
            headers += ["Total Steps", "Num Steps", "AAL"] 
            data += [total_step, num_step, aal] 
            # total_roll_back_length_error = self._model.total_roll_back_length_error 
            total_roll_back_length_error = self.model.total_roll_back_length_error 
            # errorinstance = self._model.errorinstance 
            errorinstance = self.model.errorinstance 
            totalrollbacklengtherrors = torch.tensor([total_roll_back_length_error, errorinstance], device = self.model.device) 
            dist.all_reduce(totalrollbacklengtherrors, op = dist.ReduceOp.SUM) 
            total_roll_back_length_error = totalrollbacklengtherrors[0].item() 
            errorinstance = totalrollbacklengtherrors[1].item() 
            averagerollbacklengtherror = total_roll_back_length_error / errorinstance 
            headers += ["Total Roll Back Length Error", "Error Instance", "Average Roll Back Length Error"] 
            data += [total_roll_back_length_error, errorinstance, averagerollbacklengtherror] 
            
            # tree size statistics 
            # totaltreesize = self._model.flattentreesize 
            totaltreesize = self.model.flattentreesize 
            # draftingtreesize = self._model.averagedraftingbatchsize 
            draftingtreesize = self.model.averagedraftingbatchsize 
            # totaltreesize = torch.tensor([totaltreesize, draftingtreesize], device = self.device, dtype = torch.float) 
            totaltreesize = torch.tensor([totaltreesize, draftingtreesize], device = self.model.device, dtype = torch.float) 
            dist.all_reduce(totaltreesize, op = dist.ReduceOp.SUM) 
            draftingtreesize = totaltreesize[1].item() 
            totaltreesize = totaltreesize[0].item() 
            headers += ["Effective Tree Size", "Drafting Tree Size"] 
            data += [totaltreesize/num_step, draftingtreesize/num_step] 
            
        if self.accelerator.is_main_process: 
            print(tabulate([data], headers=headers, tablefmt="grid")) 
        # self._model.updatestatistic() 
        self.model.updatestatistic() 

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name, intermediate_generations=intermediate_generations)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                self.save_json_files(generations, references, save_generations_path, f"references_{task_name}.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
