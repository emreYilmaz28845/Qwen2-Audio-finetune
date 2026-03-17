import os
import json

def generate_jsonl_file(dataset_path, output_file):
    """
    生成jsonl文件
    dataset_path: 数据集根目录路径
    output_file: 输出的jsonl文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
                
            # 读取new_label.txt文件
            label_file = os.path.join(subdir_path, "new_label.txt")
            if not os.path.exists(label_file):
                print(f"警告: 文件夹 {subdir} 中没有找到 new_label.txt 文件")
                continue
                
            # 读取标签值
            try:
                with open(label_file, 'r', encoding='utf-8') as lf:
                    label_value = float(lf.read().strip())
            except (ValueError, Exception) as e:
                print(f"错误: 无法读取文件夹 {subdir} 的标签文件: {e}")
                continue
            
            # 根据标签值确定target
            target = "抑郁" if label_value >= 53.0 else "非抑郁"
            
            # 查找所有以'_out.wav'结尾的音频文件
            for file in os.listdir(subdir_path):
                if file.endswith('_out.wav'):
                    # 获取key（与scp文件第一列格式一致）
                    key = f"{subdir}_{file.replace('_out.wav', '')}"
                    task = f"{key}_抑郁症识别"
                    
                    # 构建JSON对象
                    json_obj = {
                        "key": key,
                        "task": task,
                        "target": target
                    }
                    
                    # 写入jsonl文件
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

def generate_separate_jsonl_files(dataset_path, output_dir):
    """
    分别生成训练集和测试集的jsonl文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, "train.jsonl")
    test_output = os.path.join(output_dir, "test.jsonl")
    
    with open(train_output, 'w', encoding='utf-8') as train_f, \
         open(test_output, 'w', encoding='utf-8') as test_f:
        
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
                
            # 选择对应的输出文件
            if 't' in subdir:  # 训练集
                output_file = train_f
            elif 'v' in subdir:  # 测试集
                output_file = test_f
            else:
                continue
                
            # 读取new_label.txt文件
            label_file = os.path.join(subdir_path, "new_label.txt")
            if not os.path.exists(label_file):
                print(f"警告: 文件夹 {subdir} 中没有找到 new_label.txt 文件")
                continue
                
            # 读取标签值
            try:
                with open(label_file, 'r', encoding='utf-8') as lf:
                    label_value = float(lf.read().strip())
            except (ValueError, Exception) as e:
                print(f"错误: 无法读取文件夹 {subdir} 的标签文件: {e}")
                continue
            
            # 根据标签值确定target
            target = "抑郁" if label_value >= 53.0 else "非抑郁"
            
            # 查找所有以'_out.wav'结尾的音频文件
            for file in os.listdir(subdir_path):
                if file.endswith('_out.wav'):
                    # 获取key（与scp文件第一列格式一致）
                    key = f"{subdir}_{file.replace('_out.wav', '')}"
                    task = f"{key}_抑郁症识别"
                    
                    # 构建JSON对象
                    json_obj = {
                        "key": key,
                        "task": task,
                        "target": target
                    }
                    
                    # 写入对应的jsonl文件
                    output_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/EATD-Corpus" 
    output_dir = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd"
    
    # 生成合并的jsonl文件
    generate_jsonl_file(dataset_path, "all_data.jsonl")
    print("合并的jsonl文件已生成: all_data.jsonl")
    
    # 分别生成训练集和测试集的jsonl文件
    generate_separate_jsonl_files(dataset_path, output_dir)
    print(f"分离的jsonl文件已生成在目录: {output_dir}")
    print("训练集文件: train.jsonl")
    print("测试集文件: test.jsonl")