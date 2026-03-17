import os
import json

dataset_path = "/root/autodl-tmp/DAIC_Qwen2Audio/DAIC/data/daic_xcy/val"  
query_json_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/val_full_xcy_P.json" 
output_file = "/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/daic_multiprompt_val.jsonl"

# Read the JSON file containing query and response
with open(query_json_path, 'r', encoding='utf-8') as f:
    query_data = json.load(f)

# Create a mapping from file_id to query content
file_id_to_query = {}

for item in query_data:
    # Extract the file name from the audio tag
    audio_tag = item["query"].split("<audio>")[1].split("</audio>")[0]
    file_id = os.path.splitext(os.path.basename(audio_tag))[0]
    
    # Extract the content in the query other than the audio tag
    prompt_content = item["query"].split("</audio>")[1].strip()
    
    file_id_to_query[file_id] = prompt_content

# Generate the JSONL file
count = 0
with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_id = os.path.splitext(file)[0]
                
                if file_id in file_id_to_query:
                    json_obj = {
                        "task": f"{file_id}_抑郁症识别",
                        "prompt": f"<|audio_bos|><|AUDIO|><|audio_eos|>{file_id_to_query[file_id]}"
                    }
                    
                    outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    count += 1
                else:
                    print(f"警告: 文件 {file_id} 在query JSON文件中没有对应的记录")

print(f"JSONL文件已生成: {output_file}")
print(f"总共处理了 {count} 个音频文件")
