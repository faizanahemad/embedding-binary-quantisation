base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reg_strength = 0.2
num_epochs = 1
batch_size = 1024
lr = 0.0001

init_std = 0.01


train_modules = ['stage1', 'stage2', 'stage3', 'stage1.1'] # 'stage1', 'stage2'
save_dirs = ['run_20241125_0710', 'run_20241125_0710', 'run_20241125_0710', 'run_20241125_0710']
