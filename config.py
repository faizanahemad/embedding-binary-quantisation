base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
reg_strength = 0.1
num_epochs = 2
batch_size = 1024
lr = 0.0001
max_grad_norm = 1.0

init_std = 0.001

temperature = 10 # 10 best



epsilon=1e-8


train_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    'OneBitTwoBit',
    
] # 'stage1', 'stage2'

test_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    'OneBitTwoBit',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1755', 
    'run_20241201_1800', 
    '__', 
    'run_20241202_1422', 
]

# Thresholds are not changing with training for OneBitTwoBit