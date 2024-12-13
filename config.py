base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
reg_strength = 0.1
num_epochs = 30
batch_size = 512
lr = 0.01
max_grad_norm = 2.0

need_baselines = True
binary_baseline = False

init_std = 0.001

matryoshka_output_dim = 384//1

temperature = 10 # 10 best



epsilon=1e-8


train_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    'Matryoshka',
    
] 

test_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    'Matryoshka',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1755', 
    'run_20241201_1800', 
    '__', 
    'run_20241202_1422', 
    'run_20241213_1754', 
]

# Thresholds are not changing with training for OneBitTwoBit