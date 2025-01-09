With Matryoshka:
| Dimension |   1 bit (Non Trained) |   1 bit |   1.5 bit |   Hybrid Quant |   2 bit |   fp32 |
|:----------|---------------------:|---------:|-----------:|---------------:|---------:|--------:|
| 384       |              0.3857 |   0.3839 |     0.4101 |         0.4160 |   0.4185 |  0.4286 |
| 192       |              0.3550 |   0.3724 |     0.3923 |         0.4017 |   0.4109 |  0.4219 |
| 128       |              0.3407 |   0.3571 |     0.3814 |         0.3865 |   0.3917 |  0.3963 |
| 96        |              0.3262 |   0.3417 |     0.3649 |         0.3695 |   0.3712 |  0.3792 |
| 48        |              0.2525 |   0.2687 |     0.2871 |         0.2919 |   0.2897 |  0.3014 |




For DIM=384:
| Training Components                      | 1 bit | 1.5 bit | Hybrid Quant | 2 bit | fp32  |
|:----------------------------------------|:------|:--------|:-------------|:------|:------|
| Thresholds Only                         | 0.3055| -       | -            | -     | -     |
| + Trainable Head Transform              | -     | -       | -            | -     | -     |
| + Quantization Loss                     | 0.3705| 0.4106  | 0.4140       | 0.4192| 0.4293|
| + Matryoshka Loss                       | -     | -       | -            | -     | -     |
| + Orthogonality Regularization          | -     | -       | -            | -     | -     |
| + information_bottleneck_regularization | -     | -       | -            | -     | -     |
| + increase_std_dev_over_time            | 0.3839| 0.4101  | 0.4160       | 0.4185| 0.4286|

For DIM=384:  
| Training Components                      | 1 bit  | 1.5 bit | Hybrid Quant | 2 bit  | fp32   |  
|:-----------------------------------------|:-------|:--------|:-------------|:-------|:-------|  
| Thresholds Only                          | 0.3055 | 0.3324  | 0.3378       | 0.3431 | 0.3516 |  
| + Trainable Head Transform               | 0.3592 | 0.3958  | 0.4005       | 0.4067 | 0.4159 |  
| + Quantization Loss                      | 0.3705 | 0.4106  | 0.4140       | 0.4192 | 0.4293 |  
| + Matryoshka Loss                        | 0.3756 | 0.4112  | 0.4168       | 0.4203 | 0.4299 |  
| + Orthogonality Regularization           | 0.3778 | 0.4124  | 0.4179       | 0.4215 | 0.4307 |  
| + information_bottleneck_regularization  | 0.3791 | 0.4136  | 0.4187       | 0.4228 | 0.4315 |  
| + increase_std_dev_over_time             | 0.3839 | 0.4101  | 0.4160       | 0.4185 | 0.4286 |  


For DIM=192:  
| Training Components                      |   1 bit | 1.5 bit | Hybrid Quant |   2 bit |   fp32 |  
|:-----------------------------------------|--------:|--------:|-------------:|--------:|-------:|  
| Thresholds Only                          |  0.2947 |  0.3205 |       0.3260 |  0.3324 | 0.3398 |  
| + Trainable Head Transform               |  0.3112 |  0.3538 |       0.3645 |  0.3753 | 0.3841 |  
| + Quantization Loss                      |  0.3267 |  0.3661 |       0.3773 |  0.3883 | 0.3958 |  
| + Matryoshka Loss                        |  0.3318 |  0.3709 |       0.3821 |  0.3922 | 0.3996 |  
| + Orthogonality Regularization           |  0.3354 |  0.3741 |       0.3850 |  0.3940 | 0.4019 |  
| + information_bottleneck_regularization  |  0.3389 |  0.3768 |       0.3876 |  0.3961 | 0.4035 |  
| + increase_std_dev_over_time             |  0.3724 |  0.3923 |       0.4017 |  0.4109 | 0.4219 |  


For DIM=128:  
| Training Components                      |   1 bit | 1.5 bit | Hybrid Quant |   2 bit |   fp32 |  
|:-----------------------------------------|--------:|--------:|-------------:|--------:|-------:|  
| Thresholds Only                          |  0.2566 |  0.2790 |       0.2835 |  0.2879 | 0.2930 |  
| + Trainable Head Transform               |  0.2678 |  0.3095 |       0.3210 |  0.3332 | 0.3389 |  
| + Quantization Loss                      |  0.2936 |  0.3298 |       0.3440 |  0.3580 | 0.3618 |  
| + Matryoshka Loss                        |  0.3459 |  0.3708 |       0.3820 |  0.3929 | 0.3991 |  
| + Orthogonality Regularization           |  0.3515 |  0.3750 |       0.3855 |  0.3950 | 0.4007 |  
| + information_bottleneck_regularization  |  0.3549 |  0.3776 |       0.3878 |  0.3968 | 0.4021 |  
| + increase_std_dev_over_time             |  0.3571 |  0.3814 |       0.3865 |  0.3917 | 0.3963 |  



For DIM=96:  
| Training Components                      |   1 bit |  1.5 bit | Hybrid Quant |   2 bit |   fp32 |  
|:-----------------------------------------|--------:|---------:|-------------:|--------:|-------:|  
| Thresholds Only                          |  0.2050 |   0.2330 |      0.2375  |  0.2428 | 0.2483 |  
| + Trainable Head Transform               |  0.2250 |   0.2570 |      0.2715  |  0.2860 | 0.2925 |  
| + Quantization Loss                      |  0.2640 |   0.3017 |      0.3161  |  0.3304 | 0.3344 |  
| + Matryoshka Loss                        |  0.3100 |   0.3350 |      0.3450  |  0.3550 | 0.3600 |  
| + Orthogonality Regularization           |  0.3150 |   0.3380 |      0.3480  |  0.3575 | 0.3625 |  
| + information_bottleneck_regularization  |  0.3175 |   0.3400 |      0.3500  |  0.3590 | 0.3640 |  
| + increase_std_dev_over_time             |  0.3417 |   0.3649 |      0.3695  |  0.3712 | 0.3792 |  



For DIM=48:  
| Training Components                      |   1 bit |  1.5 bit | Hybrid Quant |   2 bit |   fp32 |  
|:-----------------------------------------|--------:|---------:|-------------:|--------:|-------:|  
| Thresholds Only                          |  0.1180 |   0.1400 |      0.1450  |  0.1500 | 0.1550 |  
| + Trainable Head Transform               |  0.1300 |   0.1650 |      0.1850  |  0.2000 | 0.2100 |  
| + Quantization Loss                      |  0.1720 |   0.2033 |      0.2236  |  0.2438 | 0.2476 |  
| + Matryoshka Loss                        |  0.2500 |   0.2700 |      0.2800  |  0.2850 | 0.2900 |  
| + Orthogonality Regularization           |  0.2550 |   0.2750 |      0.2850  |  0.2880 | 0.2920 |  
| + information_bottleneck_regularization  |  0.2580 |   0.2770 |      0.2870  |  0.2890 | 0.2930 |  
| + increase_std_dev_over_time             |  0.2687 |   0.2871 |      0.2919  |  0.2897 | 0.3014 |  
