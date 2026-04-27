import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / "../data/cmdc").resolve()
DATASETS_ROOT = Path(os.environ["AUDIOLLM_DATASETS_ROOT"]).resolve()
DATASETS_PREFIX = os.environ.get("AUDIOLLM_DATASETS_PREFIX", "../Datasets")
CMDC_DIR = DATASETS_ROOT / "CMDC"



def get_subject_folders(data_root):
    """Get and sort the HC and MDD folder lists."""
    subject_folders = [
        f for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f))
    ]

    hc_folders = sorted([f for f in subject_folders if f.startswith("HC")])
    mdd_folders = sorted([f for f in subject_folders if f.startswith("MDD")])
    
    return hc_folders, mdd_folders

def generate_fold_files(data_root, output_dir, hc_folders, mdd_folders):
    """Generate training and validation files for 5 folds."""
    
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
        
        fold_dir = os.path.join(output_dir, fold["name"])
        train_dir = os.path.join(fold_dir, "train")
        test_dir = os.path.join(fold_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Generate the training SCP file
        train_scp_path = os.path.join(train_dir, f"{fold['name']}.scp")
        with open(train_scp_path, 'w', encoding='utf-8') as f:
            # Add the MDD training set
            for idx in fold['train']['MDD']:
                if idx <= len(mdd_folders):
                    folder_name = mdd_folders[idx-1]  # Index starts from 0
                    add_audio_files(f, "MDD", folder_name, data_root)
            
            # Add the HC training set
            for idx in fold['train']['HC']:
                if idx <= len(hc_folders):
                    folder_name = hc_folders[idx-1]  # Index starts from 0
                    add_audio_files(f, "HC", folder_name, data_root)
        
        # Generate the validation SCP file
        test_scp_path = os.path.join(test_dir, f"{fold['name']}.scp")
        with open(test_scp_path, 'w', encoding='utf-8') as f:
            # Add the MDD validation set
            for idx in fold['test']['MDD']:
                if idx <= len(mdd_folders):
                    folder_name = mdd_folders[idx-1]  # Index starts from 0
                    add_audio_files(f, "MDD", folder_name, data_root)
            
            # Add the HC validation set
            for idx in fold['test']['HC']:
                if idx <= len(hc_folders):
                    folder_name = hc_folders[idx-1]  # Index starts from 0
                    add_audio_files(f, "HC", folder_name, data_root)
        
        print(f"  - 训练集: {train_scp_path}")
        print(f"  - 验证集: {test_scp_path}")

def add_audio_files(file_obj, category, folder_name, data_root):
    """Add all WAV files in the specified folder to the SCP file."""
    folder_path = os.path.join(data_root, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"警告: 路径不存在 {folder_path}")
        return
    
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            relative_path = os.path.join(
                DATASETS_PREFIX,
                os.path.relpath(file_path, start=DATASETS_ROOT),
            )
            
            # Get the filename without the extension
            file_name_without_ext = os.path.splitext(file)[0]
            
            # Generate the first column: folder name + filename
            first_column = f"{folder_name}_{file_name_without_ext}"
            
            # Write to the SCP file
            file_obj.write(f"{first_column} {relative_path}\n")

def main():
    
    
    # Get the HC and MDD folder lists
    hc_folders, mdd_folders = get_subject_folders(CMDC_DIR)
    
    print(f"找到 {len(hc_folders)} 个HC文件夹: {hc_folders}")
    print(f"找到 {len(mdd_folders)} 个MDD文件夹: {mdd_folders}")
    
    # Generate the 5-fold cross-validation files
    generate_fold_files(CMDC_DIR, OUT_DIR, hc_folders, mdd_folders)
    
    print("所有fold文件生成完成！")

if __name__ == "__main__":
    data_root = CMDC_DIR
    output_dir = OUT_DIR 
    
    # Get the HC and MDD folder lists
    hc_folders, mdd_folders = get_subject_folders(data_root)
    
    print(f"找到 {len(hc_folders)} 个HC文件夹: {hc_folders}")
    print(f"找到 {len(mdd_folders)} 个MDD文件夹: {mdd_folders}")
    
    # Generate the 5-fold cross-validation files
    generate_fold_files(data_root, output_dir, hc_folders, mdd_folders)
