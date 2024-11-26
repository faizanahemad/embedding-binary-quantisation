base_model_name = "sentence-transformers/all-mpnet-base-v2"
reg_strength = 0.1
num_epochs = 1
batch_size = 1024
lr = 0.0001

init_std = 0.01



epsilon=1e-8


train_modules = [
    # 'stage1', 
    # 'stage2', 
    # 'stage3', 
    # 'stage1.1',
    'OneBitTwoBit'
] # 'stage1', 'stage2'

test_modules = [
    # 'stage1', 
    # 'stage2', 
    # 'stage3', 
    # 'stage1.1',
    'OneBitTwoBit'
]
save_dirs = [
    'run_20241125_1450', 
    'run_20241125_1450', 
    'run_20241125_1450', 
    'run_20241125_1450', 
    'run_20241126_1651', 
]