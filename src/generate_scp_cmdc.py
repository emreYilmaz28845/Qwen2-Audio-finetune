import os
import argparse

def get_subject_folders(data_root):
    """获取HC和MDD文件夹列表并排序"""
    hc_path = os.path.join(data_root, "HC")
    mdd_path = os.path.join(data_root, "MDD")
    
    hc_folders = sorted([f for f in os.listdir(hc_path) if os.path.isdir(os.path.join(hc_path, f))])
    mdd_folders = sorted([f for f in os.listdir(mdd_path) if os.path.isdir(os.path.join(mdd_path, f))])
    
    return hc_folders, mdd_folders

def generate_fold_files(data_root, output_dir, hc_folders, mdd_folders):
    """生成5个fold的训练集和验证集文件"""
    
    folds = [
        {
            'name': 'fold1',
            'train': {'MDD': list(range(1, 21)), 'HC': list(range(1, 41))},  # 1-20, 1-40
            'test': {'MDD': list(range(21, 27)), 'HC': list(range(41, 53))}   # 21-26, 41-52
        },
        {
            'name': 'fold2', 
            'train': {'MDD': list(range(7, 27)), 'HC': list(range(13, 53))},  # 7-26, 13-52
            'test': {'MDD': list(range(1, 7)), 'HC': list(range(1, 13))}      # 1-6, 1-12
        },
        {
            'name': 'fold3',
            'train': {'MDD': list(range(13, 27)) + list(range(1, 7)),        # 13-26 & 1-6
                     'HC': list(range(25, 53)) + list(range(1, 13))},        # 25-52 & 1-12
            'test': {'MDD': list(range(7, 13)), 'HC': list(range(13, 25))}   # 7-12, 13-24
        },
        {
            'name': 'fold4',
            'train': {'MDD': list(range(19, 27)) + list(range(1, 13)),       # 19-26 & 1-12
                     'HC': list(range(34, 53)) + list(range(1, 25))},        # 34-52 & 1-24
            'test': {'MDD': list(range(13, 19)), 'HC': list(range(25, 37))}  # 13-18, 25-36
        },
        {
            'name': 'fold5',
            'train': {'MDD': list(range(25, 27)) + list(range(1, 19)),       # 25-26 & 1-18
                     'HC': list(range(49, 53)) + list(range(1, 37))},        # 49-52 & 1-36
            'test': {'MDD': list(range(19, 25)), 'HC': list(range(37, 49))}  # 19-24, 37-48
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fold in folds:
        print(f"生成 {fold['name']} 的文件...")
        
        # 生成训练集scp文件
        train_scp_path = os.path.join(output_dir, f"{fold['name']}_train.scp")
        with open(train_scp_path, 'w', encoding='utf-8') as f:
            # 添加MDD训练集
            for idx in fold['train']['MDD']:
                if idx <= len(mdd_folders):
                    folder_name = mdd_folders[idx-1]  # 索引从0开始
                    add_audio_files(f, "MDD", folder_name, data_root)
            
            # 添加HC训练集
            for idx in fold['train']['HC']:
                if idx <= len(hc_folders):
                    folder_name = hc_folders[idx-1]  # 索引从0开始
                    add_audio_files(f, "HC", folder_name, data_root)
        
        # 生成验证集scp文件
        test_scp_path = os.path.join(output_dir, f"{fold['name']}_val.scp")
        with open(test_scp_path, 'w', encoding='utf-8') as f:
            # 添加MDD验证集
            for idx in fold['test']['MDD']:
                if idx <= len(mdd_folders):
                    folder_name = mdd_folders[idx-1]  # 索引从0开始
                    add_audio_files(f, "MDD", folder_name, data_root)
            
            # 添加HC验证集
            for idx in fold['test']['HC']:
                if idx <= len(hc_folders):
                    folder_name = hc_folders[idx-1]  # 索引从0开始
                    add_audio_files(f, "HC", folder_name, data_root)
        
        print(f"  - 训练集: {train_scp_path}")
        print(f"  - 验证集: {test_scp_path}")

def add_audio_files(file_obj, category, folder_name, data_root):
    """将指定文件夹中的所有wav文件添加到scp文件中"""
    folder_path = os.path.join(data_root, category, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"警告: 路径不存在 {folder_path}")
        return
    
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            absolute_path = os.path.abspath(file_path)
            
            # 获取文件名（不带后缀）
            file_name_without_ext = os.path.splitext(file)[0]
            
            # 生成第一列：文件夹名_文件名
            first_column = f"{folder_name}_{file_name_without_ext}"
            
            # 写入scp文件
            file_obj.write(f"{first_column} {absolute_path}\n")

def main():
    parser = argparse.ArgumentParser(description='生成分层5折交叉验证的scp文件')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据集根目录路径，包含HC和MDD文件夹')
    parser.add_argument('--output_dir', type=str, default='kfold_splits',
                       help='输出目录路径，默认为kfold_splits')
    
    args = parser.parse_args()
    
    # 检查数据集路径是否存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据集路径 '{args.data_root}' 不存在")
        return
    
    # 获取HC和MDD文件夹列表
    hc_folders, mdd_folders = get_subject_folders(args.data_root)
    
    print(f"找到 {len(hc_folders)} 个HC文件夹: {hc_folders}")
    print(f"找到 {len(mdd_folders)} 个MDD文件夹: {mdd_folders}")
    
    # 生成5折交叉验证文件
    generate_fold_files(args.data_root, args.output_dir, hc_folders, mdd_folders)
    
    print("所有fold文件生成完成！")

if __name__ == "__main__":
    data_root = "/root/autodl-tmp/CMDC_data"
    output_dir = "/root/autodl-tmp/Qwen2-Audio-finetune/data/cmdc"
    
    # 获取HC和MDD文件夹列表
    hc_folders, mdd_folders = get_subject_folders(data_root)
    
    print(f"找到 {len(hc_folders)} 个HC文件夹: {hc_folders}")
    print(f"找到 {len(mdd_folders)} 个MDD文件夹: {mdd_folders}")
    
    # 生成5折交叉验证文件
    generate_fold_files(data_root, output_dir, hc_folders, mdd_folders)