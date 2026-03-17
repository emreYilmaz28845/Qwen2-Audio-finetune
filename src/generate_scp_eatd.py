import os
import glob

def generate_scp_files(dataset_path, output_dir):
    """
    分别生成训练集和测试集的scp文件
    dataset_path: 数据集根目录路径
    output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 分别处理训练集和测试集
    train_output = os.path.join(output_dir, "train.scp")
    test_output = os.path.join(output_dir, "test.scp")
    
    with open(train_output, 'w', encoding='utf-8') as train_f, \
         open(test_output, 'w', encoding='utf-8') as test_f:
        
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
                
            # 根据文件夹名称判断是训练集还是测试集
            if 't' in subdir:  # 训练集
                output_file = train_f
                set_type = "train"
            elif 'v' in subdir:  # 测试集
                output_file = test_f
                set_type = "test"
            else:
                continue  # 跳过不包含t或v的文件夹
            
            # 查找所有以'_out.wav'结尾的音频文件
            wav_pattern = os.path.join(subdir_path, '*_out.wav')
            wav_files = glob.glob(wav_pattern)
            
            print(f"处理{set_type}集文件夹: {subdir}, 找到{len(wav_files)}个音频文件")
            
            for wav_file in wav_files:
                # 获取文件名（不带后缀）
                filename = os.path.basename(wav_file).replace('.wav', '')
                # 组合第一列：子文件夹名字_文件名（不带_out）
                first_column = f"{subdir}_{filename.replace('_out', '')}"
                # 第二列：音频文件的绝对路径
                absolute_path = os.path.abspath(wav_file)
                
                # 写入对应的scp文件
                output_file.write(f"{first_column} {absolute_path}\n")

if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/EATD-Corpus" 
    output_dir = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd" 
    
    generate_scp_files(dataset_path, output_dir)
    print(f"SCP文件已生成在目录: {output_dir}")
    print("训练集文件: train.scp")
    print("测试集文件: test.scp")