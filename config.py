base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
reg_strength = 0.1
num_epochs = 2
batch_size = 1024
lr = 0.0001

init_std = 0.01

temperature = 10



epsilon=1e-8


train_modules = [
    # 'stage1', 
    'stage2', 
    # 'stage3', 
    # 'stage1.1',
    # 'OneBitTwoBit',
    
] # 'stage1', 'stage2'

test_modules = [
    # 'stage1', 
    'stage2', 
    # 'stage3', 
    # 'stage1.1',
    # 'OneBitTwoBit',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1638', 
    'run_20241130_2132', 
    'run_20241130_2132', 
    'run_20241130_2132', 
]