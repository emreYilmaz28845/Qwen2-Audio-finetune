import os
import csv
import json

dataset_path = "/root/autodl-tmp/DAIC_Qwen2Audio/DAIC/data/daic_xcy/val"  
csv_file_path = "/root/autodl-tmp/DAIC_Qwen2Audio/DAIC/label_csv/dev_split_Depression_AVEC2017.csv"  
output_file = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/daic_multitask_val.jsonl"

# Read the CSV file and build the mapping
participant_to_label = {}

with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        participant_id = row['Participant_ID']
        phq8_binary = row['PHQ8_Binary']
        
        if phq8_binary == '1':
            label = "抑郁"
        elif phq8_binary == '0':
            label = "非抑郁"
        else:
            print(f"警告: {participant_id} 的PHQ8_Binary值异常: {phq8_binary}")
            continue
        
        participant_to_label[participant_id] = label

# Generate the JSONL file
count = 0
with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                subfolder_name = os.path.basename(root)
                
                if subfolder_name in participant_to_label:
                    file_id = os.path.splitext(file)[0]
                    label = participant_to_label[subfolder_name]
                    
                    json_obj = {
                        "key": file_id,
                        "task": f"{file_id}_抑郁症识别",
                        "target": label
                    }
                    
                    outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    count += 1
                else:
                    print(f"警告: 文件夹 {subfolder_name} 在CSV中无记录")

print(f"JSONL文件已生成: {output_file}")
print(f"总共处理了 {count} 个音频文件")
