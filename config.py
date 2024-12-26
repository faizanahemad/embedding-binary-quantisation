base_model_name = "jinaai/jina-embeddings-v3"# "", "sentence-transformers/all-MiniLM-L12-v2"
base_model_name = "sentence-transformers/all-MiniLM-L12-v2"
reg_strength = 0.01
num_epochs = 5
batch_size = 512
lr = 0.01 # 0.001
max_grad_norm = 1.0

max_samples_per_dataset = 10000 # 10000

need_baselines = True
binary_baseline = False

dimension_levels = [8, 4, 2, 1]

use_information_bottleneck = False

use_information_bottleneck_regularization = False
increase_std_dev_over_time = True
quantization_regularization = True

use_orthogonality_regularization = True

enable_matryoshka_training = True


use_rms_norm = False

customized_matryoshka_output_dim = 64
customized_matryoshka_kwargs = dict(two_bits=32, one_and_half_bits=32, one_bits=128, half_bits=192, expand=True)


init_std = 0.01

matryoshka_output_dim = 48

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
    # 'Matryoshka_1bit',
    # 'Matryoshka_2bit_3bit',
    # 'Matryoshka_1_5bit',
    'CustomizedMatryoshka',
] 

test_modules = [
    # 'stage1', 
    # 'stage1.1',
    # 'stage2', 
    # 'stage3', 
    # 'OneBitTwoBit',
    
    # 'Matryoshka',
    'Matryoshka_2bit',
    # 'Matryoshka_1bit',
    # 'Matryoshka_2bit_3bit',
    'Matryoshka_1_5bit',
    'CustomizedMatryoshka',
]
save_dirs = [
    'run_20241201_1638', 
    'run_20241201_1755', 
    'run_20241201_1800', 
    '__', 
    'run_20241202_1422', 
    "run_20241216_1750",
    
    'run_20241222_0631', 
    'run_20241221_0309',
    'run_20241222_0631',
    
    'run_20241222_0646',
    
    'run_20241223_1748',
]

# Thresholds are not changing with training for OneBitTwoBit