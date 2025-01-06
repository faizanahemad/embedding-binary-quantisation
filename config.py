base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
# base_model_name = "nomic-ai/modernbert-embed-base"
nomic_ai_prefix_need = False
reg_strength = 0.01
num_epochs = 1
batch_size = 2048
lr = 0.01 # 0.001
max_grad_norm = 1.0

large_task_list = False

max_samples_per_dataset = 10000 # 10000

need_baselines = False
binary_baseline = False # QuantStage1_Untrained

dimension_levels = [8, 4, 3, 2, 1]

use_information_bottleneck = False
use_information_bottleneck_regularization = False
increase_std_dev_over_time = True
quantization_regularization = True
use_orthogonality_regularization = True
use_rms_norm = False

# Important
enable_matryoshka_training = True
matryoshka_output_dim = 384//8
# Important

customized_matryoshka_output_dim = 384
customized_matryoshka_kwargs = dict(two_bits=64, one_and_half_bits=64, one_bits=128, half_bits=192, expand=True)


init_std = 0.01



temperature = 10 # 10 best



epsilon=1e-8


train_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    # 'Matryoshka',
    'Matryoshka_2bit',
    'Matryoshka_1bit',
    # 'Matryoshka_2bit_3bit',
    'Matryoshka_1_5bit',
    # 'CustomizedMatryoshka',
] 

test_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    
    # 'Matryoshka',
    'Matryoshka_2bit',
    'Matryoshka_1bit',
    'Matryoshka_2bit_3bit',
    'Matryoshka_1_5bit',
    # 'CustomizedMatryoshka',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1755', 
    'run_20241201_1800', 
    '__', 
    'run_20241202_1422', 
    
    
    
    "run_20250104_0423",
    
    'run_20250105_0543', # 2 bit
     
    'run_20250105_0543', # 1 bit
    'run_20250105_0543', # 2 bit to 3 bit
    
    'run_20250105_0543', # 1.5 bit
    
    'run_20250103_1018',
]



# Thresholds are not changing with training for OneBitTwoBit

# 2 bit -  run_20250103_1019
# 2 bit Non Matryoshka - run_20250104_0740



# 1 bit - run_20250104_1123
# 1 bit Non Matryoshka - run_20250104_1118

# 1.5 bit - run_20250104_1125
# 1.5 bit Non Matryoshka - run_20250104_1119


# MiniLM
# Matryoska enabled = run_20250105_0543
# Matryoska disabled = run_20250105_0856

