base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reg_strength = 0.1
num_epochs = 3
batch_size = 512
lr = 0.0001


train_modules = ['stage1', 'stage2', 'stage3', 'stage1.1'] # 'stage1', 'stage2'
save_dirs = ['run_20241124_2031', 'run_20241124_2031', 'run_20241124_2031', 'run_20241124_2031']
