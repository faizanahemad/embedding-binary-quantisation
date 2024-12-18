base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
reg_strength = 0.002
num_epochs = 3
batch_size = 512
lr = 0.001 # 0.001
max_grad_norm = 1.0

max_samples_per_dataset = 10000 # 10000

need_baselines = True
binary_baseline = False

dimension_levels = [8, 4, 2, 1]



init_std = 0.01

matryoshka_output_dim = 384

temperature = 10 # 10 best



epsilon=1e-8


train_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    # 'Matryoshka',
    # 'Matryoshka_2bit',
    'Matryoshka_1bit',
    # 'Matryoshka_2bit_3bit',
    
] 

test_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    
    # 'Matryoshka',
    # 'Matryoshka_2bit',
    'Matryoshka_1bit',
    # 'Matryoshka_2bit_3bit',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1755', 
    'run_20241201_1800', 
    '__', 
    'run_20241202_1422', 
    "run_20241216_1750",
    'run_20241218_1707', 
    'run_20241219_1053',
    'run_20241218_1707',
]

# Thresholds are not changing with training for OneBitTwoBit