import os
import shutil

def restructure():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'data', 'raw', 'dataset')
    
    # Target standardized directories
    target_dirs = {
        'healthy': os.path.join(dataset_dir, 'healthy_npk'),
        'nitrogen': os.path.join(dataset_dir, 'nitrogen_N_deficiency'),
        'phosphorus': os.path.join(dataset_dir, 'phosphorus_P_deficiency'),
        'potas': os.path.join(dataset_dir, 'potassium_K_deficiency')
    }
    
    for t_dir in target_dirs.values():
        os.makedirs(t_dir, exist_ok=True)

    splits = ['train', 'test', 'validation']
    moved = 0
    
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            continue
            
        # Iterate over subdirectories within the split
        for sub_dir_name in os.listdir(split_path):
            sub_dir_path = os.path.join(split_path, sub_dir_name)
            if not os.path.isdir(sub_dir_path):
                continue
                
            # Determine target 
            name_lower = sub_dir_name.lower()
            target_key = None
            if 'healthy' in name_lower:
                target_key = 'healthy'
            elif 'nitrogen' in name_lower or '-n' in name_lower:
                target_key = 'nitrogen'
            elif 'phos' in name_lower or '-p' in name_lower:
                target_key = 'phosphorus'
            elif 'potas' in name_lower or '-k' in name_lower or '=k' in name_lower:
                target_key = 'potas'
                
            if target_key:
                target_folder = target_dirs[target_key]
                for file_name in os.listdir(sub_dir_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src = os.path.join(sub_dir_path, file_name)
                        # To avoid naming collisions, prepend the split name and original folder name
                        unique_name = f"{split}_{sub_dir_name}_{file_name}"
                        dst = os.path.join(target_folder, unique_name)
                        shutil.move(src, dst)
                        moved += 1

    print(f"Restructure Complete! Successfully flattened {moved} raw images.")
    
    # Clean up empty split directories
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            try:
                shutil.rmtree(split_path)
                print(f"Cleaned up {split} directory.")
            except Exception as e:
                pass

if __name__ == "__main__":
    restructure()
