import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
class prepare_multitaskprompt:
    def __init__(self,model_path=None):
        self.data = []
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def load(self,in_file):
        with open(in_file) as f:
            for line in f:
                self.data.append(json.loads(line))
    def append(self,prompt,task_name="ASR"):
        # messages = [
        #     {"role": "system", "content": self.prompt},
        #     {"role": "user", "content": target}
        # ]
        item = {"task":task_name,"prompt":prompt}
        self.data.append(item)
    def save(self,out_file):
        with open(out_file,"w") as f:
            for item in self.data:
                write_item = "{"
                for i in item:
                    write_item+='"'+ i +'": "' + item[i]+'", '
                write_item = write_item.strip(", ")
                write_item+="}\n"
                f.write(rf"{write_item}")


if __name__ == "__main__":
    prep = prepare_multitaskprompt()
    prep.append("Transcribe speech to text.","ASR")
    prep.append("Transcribe speech to English.","ZH2EN")
    prep.append("Transcribe speech to Chinese.","EN2ZH")
    prep.append("Transcribe speech to text, below are the previous historical transcription texts:{}.","prevtext")
    prep.append("Transcribe speech to text, follow words may occur in audio:{}.","hotword")
    # /hpc_stor01/home/yangui.fang_sx/workingspace/data
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multiprompt.jsonl")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/train/multiprompt.jsonl")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/dev/multiprompt.jsonl")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/test/multiprompt.jsonl")
