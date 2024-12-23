

Main Score Results:
| Task      |   Matryoshka_1bit_Trained |   Matryoshka_1bit_Untrained |   Matryoshka_1bit_non_quantized |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|--------------------------:|----------------------------:|--------------------------------:|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                    0.4193 |                      0.42   |                          0.4659 |                         0.4521 |                    0.4547 |                          0.4687 |     0.4714 |
| NFCorpus  |                    0.2732 |                      0.2724 |                          0.32   |                         0.2974 |                    0.306  |                          0.3191 |     0.3225 |
| SCIDOCS   |                    0.18   |                      0.1858 |                          0.2175 |                         0.2039 |                    0.2048 |                          0.2186 |     0.2182 |
| SciFact   |                    0.5536 |                      0.5689 |                          0.6194 |                         0.605  |                    0.6052 |                          0.6206 |     0.6264 |
| TRECCOVID |                    0.5007 |                      0.4743 |                          0.5253 |                         0.529  |                    0.5125 |                          0.5228 |     0.5082 |
| Total     |                    0.3854 |                      0.3843 |                          0.4296 |                         0.4175 |                    0.4166 |                          0.43   |     0.4293 |


Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|----------------------------:|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                      0.4405 |                         0.4466 |                    0.4547 |                          0.4547 |     0.4714 |
| NFCorpus  |                      0.2892 |                         0.3009 |                    0.304  |                          0.304  |     0.3225 |
| SCIDOCS   |                      0.1958 |                         0.2036 |                    0.2048 |                          0.2048 |     0.2182 |
| SciFact   |                      0.5934 |                         0.6037 |                    0.6014 |                          0.6014 |     0.6264 |
| TRECCOVID |                      0.5178 |                         0.5404 |                    0.5371 |                          0.5371 |     0.5082 |
| Total     |                      0.4073 |                         0.419  |                    0.4204 |                          0.4204 |     0.4293 |

Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|----------------------------:|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                      0.4405 |                         0.4466 |                    0.4547 |                          0.4547 |     0.4714 |
| NFCorpus  |                      0.2892 |                         0.3009 |                    0.304  |                          0.304  |     0.3225 |
| SCIDOCS   |                      0.1958 |                         0.2036 |                    0.2048 |                          0.2048 |     0.2182 |
| SciFact   |                      0.5934 |                         0.6037 |                    0.6014 |                          0.6014 |     0.6264 |
| TRECCOVID |                      0.5178 |                         0.5404 |                    0.5371 |                          0.5371 |     0.5082 |
| Total     |                      0.4073 |                         0.419  |                    0.4204 |                          0.4204 |     0.4293 |


Orthogonality Regularization
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4466 |                    0.4547 |                          0.4689 |     0.4714 |
| NFCorpus  |                         0.3009 |                    0.304  |                          0.3196 |     0.3225 |
| SCIDOCS   |                         0.2036 |                    0.2048 |                          0.219  |     0.2182 |
| SciFact   |                         0.6037 |                    0.6014 |                          0.6203 |     0.6264 |
| TRECCOVID |                         0.5404 |                    0.5371 |                          0.5231 |     0.5082 |
| Total     |                         0.419  |                    0.4204 |                          0.4302 |     0.4293 |

GELU + RMSNorm
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4487 |                    0.4557 |                          0.4692 |     0.4714 |
| NFCorpus  |                         0.3018 |                    0.3044 |                          0.3198 |     0.3225 |
| SCIDOCS   |                         0.2028 |                    0.2033 |                          0.2186 |     0.2182 |
| SciFact   |                         0.6021 |                    0.6057 |                          0.62   |     0.6264 |
| TRECCOVID |                         0.5348 |                    0.5258 |                          0.523  |     0.5082 |
| Total     |                         0.418  |                    0.419  |                          0.4301 |     0.4293 |


GELU + Multiple layers + extra training

Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4451 |                    0.4531 |                          0.4706 |     0.4714 |
| NFCorpus  |                         0.3021 |                    0.308  |                          0.3195 |     0.3225 |
| SCIDOCS   |                         0.2054 |                    0.206  |                          0.2184 |     0.2182 |
| SciFact   |                         0.5917 |                    0.5854 |                          0.618  |     0.6264 |
| TRECCOVID |                         0.4981 |                    0.5038 |                          0.5165 |     0.5082 |
| Total     |                         0.4085 |                    0.4113 |                          0.4286 |     0.4293 |



GELU Results:
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4485 |                    0.457  |                          0.4669 |     0.4714 |
| NFCorpus  |                         0.2992 |                    0.3061 |                          0.3186 |     0.3225 |
| SCIDOCS   |                         0.2045 |                    0.205  |                          0.2186 |     0.2182 |
| SciFact   |                         0.6094 |                    0.6126 |                          0.6205 |     0.6264 |
| TRECCOVID |                         0.5336 |                    0.5201 |                          0.5241 |     0.5082 |
| Total     |                         0.419  |                    0.4202 |                          0.4297 |     0.4293 |


GELU + LN
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4434 |                    0.4474 |                          0.4644 |     0.4714 |
| NFCorpus  |                         0.2976 |                    0.3021 |                          0.3125 |     0.3225 |
| SCIDOCS   |                         0.2004 |                    0.2006 |                          0.2138 |     0.2182 |
| SciFact   |                         0.6056 |                    0.6147 |                          0.6222 |     0.6264 |
| TRECCOVID |                         0.5149 |                    0.5204 |                          0.5075 |     0.5082 |
| Total     |                         0.4124 |                    0.417  |                          0.4241 |     0.4293 |


GELU + LN + layers
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4397 |                    0.4427 |                          0.4646 |     0.4714 |
| NFCorpus  |                         0.2874 |                    0.2912 |                          0.3136 |     0.3225 |
| SCIDOCS   |                         0.2008 |                    0.2042 |                          0.2125 |     0.2182 |
| SciFact   |                         0.5802 |                    0.587  |                          0.6076 |     0.6264 |
| TRECCOVID |                         0.4834 |                    0.4924 |                          0.5118 |     0.5082 |
| Total     |                         0.3983 |                    0.4035 |                          0.422  |     0.4293 |


GELU + Multi-Layers
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4468 |                    0.4509 |                          0.4681 |     0.4714 |
| NFCorpus  |                         0.3018 |                    0.3034 |                          0.3192 |     0.3225 |
| SCIDOCS   |                         0.204  |                    0.2069 |                          0.2158 |     0.2182 |
| SciFact   |                         0.5812 |                    0.5849 |                          0.613  |     0.6264 |
| TRECCOVID |                         0.4905 |                    0.4935 |                          0.5171 |     0.5082 |
| Total     |                         0.4049 |                    0.4079 |                          0.4266 |     0.4293 |

Additional MSE loss on probabilities:
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4522 |                    0.4546 |                          0.4687 |     0.4714 |
| NFCorpus  |                         0.2977 |                    0.3056 |                          0.3191 |     0.3225 |
| SCIDOCS   |                         0.204  |                    0.2049 |                          0.2186 |     0.2182 |
| SciFact   |                         0.605  |                    0.6052 |                          0.6206 |     0.6264 |
| TRECCOVID |                         0.5285 |                    0.5125 |                          0.5241 |     0.5082 |
| Total     |                         0.4175 |                    0.4166 |                          0.4302 |     0.4293 |


Main Score Results:
| Task      |   Matryoshka_1bit_Trained |   Matryoshka_1bit_Untrained |   Matryoshka_1bit_non_quantized |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|--------------------------:|----------------------------:|--------------------------------:|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                    0.4193 |                      0.42   |                          0.4659 |                         0.4313 |                    0.4377 |                          0.4401 |     0.4714 |
| NFCorpus  |                    0.2732 |                      0.2724 |                          0.32   |                         0.2947 |                    0.3009 |                          0.3032 |     0.3225 |
| SCIDOCS   |                    0.18   |                      0.1858 |                          0.2175 |                         0.1971 |                    0.1996 |                          0.2016 |     0.2182 |
| SciFact   |                    0.5536 |                      0.5689 |                          0.6194 |                         0.5673 |                    0.583  |                          0.5926 |     0.6264 |
| TRECCOVID |                    0.5007 |                      0.4743 |                          0.5253 |                         0.4789 |                    0.4852 |                          0.4865 |     0.5082 |
| Total     |                    0.3854 |                      0.3843 |                          0.4296 |                         0.3939 |                    0.4013 |                          0.4048 |     0.4293 |


Without any transformation:
Main Score Results:
| Task      |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Matryoshka_2bit_non_quantized |   Original |
|:----------|-------------------------------:|--------------------------:|--------------------------------:|-----------:|
| ArguAna   |                         0.4551 |                    0.461  |                          0.4714 |     0.4714 |
| NFCorpus  |                         0.3075 |                    0.3107 |                          0.3225 |     0.3225 |
| SCIDOCS   |                         0.2115 |                    0.2117 |                          0.2182 |     0.2182 |
| SciFact   |                         0.6132 |                    0.6114 |                          0.6264 |     0.6264 |
| TRECCOVID |                         0.4934 |                    0.5054 |                          0.5082 |     0.5082 |
| Total     |                         0.4161 |                    0.42   |                          0.4293 |     0.4293 |


| Total     |                    0.3854 |                      0.3843 |                          0.4296 |                         0.4175 |                    0.4166 |                          0.43   |     0.4293 |
| Total     |                    0.3854 |                      0.3843 |                          0.4296 |                         0.3939 |                    0.4013 |                          0.4048 |     0.4293 |
| Total     |                    0.3854 |                      0.3843 |                          0.4296 |                         0.4021 |                    0.4039 |                          0.418  |     0.4293 |


- Multiple layers did not work
- One GELU Non linearity worked
- Layer normalization did not work