| Dimension | 1 bit (Non Trained for Quant, Non Matryoshka) | 1.5 bit (Non Trained for Quant, Non Matryoshka) | Hybrid Quant (Non Trained for Quant, Non Matryoshka) | 2 bit (Non Trained for Quant, Non Matryoshka) | 1 bit (Non Trained for Quant) | 1.5 bit (Non Trained for Quant) | Hybrid Quant (Non Trained for Quant) | 2 bit (Non Trained for Quant) | fp32   | 1 bit (Matryoshka) | 1.5 bit (Matryoshka) | Hybrid Quant (Matryoshka) | 2 bit (Matryoshka) | fp32 (Matryoshka) |
|:----------|----------------------------------------------:|------------------------------------------------:|-----------------------------------------------------:|----------------------------------------------:|------------------------------:|--------------------------------:|-------------------------------------:|------------------------------:|-------:|-------------------:|---------------------:|--------------------------:|-------------------:|------------------:|
| 768       |                                        0.4350 |                                          0.4610 |                                               0.4633 |                                        0.4640 |                        0.4370 |                          0.4639 |                               0.4660 |                        0.4664 | 0.4720 |             0.4381 |               0.4536 |                    0.4680 |             0.4687 |            0.4720 |
| 384       |                                        0.3900 |                                          0.4250 |                                               0.4334 |                                        0.4370 |                        0.3998 |                          0.4318 |                               0.4399 |                        0.4425 | 0.4632 |             0.4167 |               0.4429 |                    0.4509 |             0.4593 |            0.4695 |
| 256       |                                        0.3300 |                                          0.4000 |                                               0.4150 |                                        0.4200 |                        0.3485 |                          0.4165 |                               0.4248 |                        0.4296 | 0.4559 |             0.3691 |               0.4228 |                    0.4465 |             0.4513 |            0.4680 |
| 192       |                                        0.2900 |                                          0.3550 |                                               0.3745 |                                        0.3850 |                        0.3107 |                          0.3732 |                               0.3810 |                        0.4050 | 0.4405 |             0.3285 |               0.3901 |                    0.4245 |             0.4327 |            0.4512 |
| 96        |                                        0.2500 |                                          0.3150 |                                               0.3395 |                                        0.3500 |                        0.2750 |                          0.3294 |                               0.3440 |                        0.3671 | 0.4030 |             0.2908 |               0.3455 |                    0.3850 |             0.3919 |            0.4247 |



For DIM=768:  
| Training Components                      | 1 bit   | 1.5 bit | Hybrid Quant | 2 bit   | fp32   |  
|:-----------------------------------------|:--------|:--------|:-------------|:--------|:-------|  
| Thresholds Only                          | 0.4350  | 0.4610  | 0.4631       | 0.4640  | 0.4665 |  
| + Trainable Head Transform               | 0.4372  | 0.4635  | 0.4658       | 0.4665  | 0.4685 |  
| + Quantization Loss                      | 0.4390  | 0.4658  | 0.4680       | 0.4685  | 0.4700 |  
| + Matryoshka Loss                        | 0.4370  | 0.4639  | 0.4655       | 0.4664  | 0.4720 |  
| + Orthogonality Regularization           | 0.4385  | 0.4659  | 0.4676       | 0.4682  | 0.4735 |  
| + Information Bottleneck Regularization  | 0.4398  | 0.4676  | 0.4694       | 0.4698  | 0.4747 |  
| + Increase Std Dev Over Time             | 0.4381  | 0.4536  | 0.4670       | 0.4687  | 0.4720 |  


For DIM=384:  
| Training Components                      | 1 bit   | 1.5 bit | Hybrid Quant | 2 bit   | fp32   |  
|:-----------------------------------------|:--------|:--------|:-------------|:--------|:-------|  
| Thresholds Only                          | 0.3900  | 0.4250  | 0.4358       | 0.4370  | 0.4395 |  
| + Trainable Head Transform               | 0.3922  | 0.4275  | 0.4380       | 0.4390  | 0.4410 |  
| + Quantization Loss                      | 0.3945  | 0.4298  | 0.4402       | 0.4415  | 0.4430 |  
| + Matryoshka Loss                        | 0.3998  | 0.4318  | 0.4410       | 0.4425  | 0.4632 |  
| + Orthogonality Regularization           | 0.4012  | 0.4329  | 0.4430       | 0.4440  | 0.4645 |  
| + Information Bottleneck Regularization  | 0.4025  | 0.4340  | 0.4442       | 0.4455  | 0.4658 |  
| + Increase Std Dev Over Time             | 0.4167  | 0.4429  | 0.4565       | 0.4593  | 0.4695 |  




For DIM=256:  
| Training Components                      | 1 bit   | 1.5 bit | Hybrid Quant | 2 bit   | fp32   |  
|:-----------------------------------------|:--------|:--------|:-------------|:--------|:-------|  
| Thresholds Only                          | 0.3300  | 0.4000  | 0.4135       | 0.4200  | 0.4230 |  
| + Trainable Head Transform               | 0.3324  | 0.4025  | 0.4157       | 0.4223  | 0.4249 |  
| + Quantization Loss                      | 0.3345  | 0.4047  | 0.4181       | 0.4245  | 0.4269 |  
| + Matryoshka Loss                        | 0.3485  | 0.4165  | 0.4252       | 0.4296  | 0.4559 |  
| + Orthogonality Regularization           | 0.3503  | 0.4184  | 0.4271       | 0.4314  | 0.4576 |  
| + Information Bottleneck Regularization  | 0.3520  | 0.4202  | 0.4290       | 0.4333  | 0.4594 |  
| + Increase Std Dev Over Time             | 0.3691  | 0.4228  | 0.4420       | 0.4513  | 0.4680 |  


For DIM=192:  
| Training Components                      | 1 bit   | 1.5 bit | Hybrid Quant | 2 bit   | fp32   |  
|:-----------------------------------------|:--------|:--------|:-------------|:--------|:-------|  
| Thresholds Only                          | 0.2900  | 0.3550  | 0.3755       | 0.3850  | 0.3880 |  
| + Trainable Head Transform               | 0.2922  | 0.3575  | 0.3778       | 0.3872  | 0.3900 |  
| + Quantization Loss                      | 0.2943  | 0.3597  | 0.3801       | 0.3893  | 0.3920 |  
| + Matryoshka Loss                        | 0.3107  | 0.3732  | 0.3945       | 0.4050  | 0.4405 |  
| + Orthogonality Regularization           | 0.3125  | 0.3752  | 0.3964       | 0.4068  | 0.4422 |  
| + Information Bottleneck Regularization  | 0.3142  | 0.3770  | 0.3983       | 0.4087  | 0.4440 |  
| + Increase Std Dev Over Time             | 0.3285  | 0.3901  | 0.4185       | 0.4327  | 0.4512 |  




For DIM=96:  
| Training Components                      | 1 bit   | 1.5 bit | Hybrid Quant | 2 bit   | fp32   |  
|:-----------------------------------------|:--------|:--------|:-------------|:--------|:-------|  
| Thresholds Only                          | 0.2500  | 0.3150  | 0.3386       | 0.3500  | 0.3530 |  
| + Trainable Head Transform               | 0.2522  | 0.3174  | 0.3408       | 0.3521  | 0.3549 |  
| + Quantization Loss                      | 0.2543  | 0.3196  | 0.3431       | 0.3542  | 0.3568 |  
| + Matryoshka Loss                        | 0.2750  | 0.3294  | 0.3548       | 0.3671  | 0.4030 |  
| + Orthogonality Regularization           | 0.2768  | 0.3314  | 0.3567       | 0.3689  | 0.4047 |  
| + Information Bottleneck Regularization  | 0.2785  | 0.3332  | 0.3586       | 0.3708  | 0.4065 |  
| + Increase Std Dev Over Time             | 0.2908  | 0.3455  | 0.3767       | 0.3919  | 0.4247 |  
