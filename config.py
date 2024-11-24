base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reg_strength = 0.05
num_epochs = 10
batch_size = 512


train_modules = ['stage1', 'stage2', 'stage3', 'stage1.1'] # 'stage1', 'stage2'
save_dirs = ['run_20241124_0454', 'run_20241124_1052', 'run_20241124_0454', 'run_20241124_1052']
