
Dim=384
Main Score Results:
| Task                    |   Original |   QuantStage1_Untrained |
|:------------------------|-----------:|------------------------:|
| ArguAna                 |     0.4714 |                  0.4166 |
| CQADupstackTexRetrieval |     0.3166 |                  0.264  |
| ClimateFEVER            |     0.2157 |                  0.1494 |
| DBPedia                 |     0.3336 |                  0.2766 |
| FEVER                   |     0.5591 |                  0.441  |
| FiQA2018                |     0.3727 |                  0.3166 |
| HotpotQA                |     0.4459 |                  0.3211 |
| MSMARCO                 |     0.6742 |                  0.6256 |
| NFCorpus                |     0.3225 |                  0.2831 |
| NQ                      |     0.4647 |                  0.3961 |
| QuoraRetrieval          |     0.8776 |                  0.8567 |
| SCIDOCS                 |     0.2182 |                  0.1955 |
| SciFact                 |     0.6264 |                  0.5672 |
| TRECCOVID               |     0.5082 |                  0.466  |
| Touche2020              |     0.1722 |                  0.1559 |
| Total                   |     0.4386 |                  0.3821 |

DIM=384
Without Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Original |   QuantStage1_Untrained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|-----------:|------------------------:|
| ArguAna   |                      0.4432 |                    0.4146 |                         0.4543 |                    0.4598 |     0.4714 |                  0.4166 |
| NFCorpus  |                      0.3018 |                    0.2803 |                         0.3075 |                    0.3114 |     0.3225 |                  0.2831 |
| SCIDOCS   |                      0.204  |                    0.1934 |                         0.2117 |                    0.2118 |     0.2182 |                  0.1955 |
| SciFact   |                      0.6023 |                    0.5603 |                         0.6109 |                    0.6114 |     0.6264 |                  0.5672 |
| TRECCOVID |                      0.5017 |                    0.4756 |                         0.4995 |                    0.5017 |     0.5082 |                  0.466  |
| Total     |                      0.4106 |                    0.3848 |                         0.4168 |                    0.4192 |     0.4293 |                  0.3857 |

DIM=384
With Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |   Original |   QuantStage1_Untrained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|-----------:|------------------------:|
| ArguAna   |                      0.446  |                    0.4052 |                         0.4464 |                    0.4532 |     0.4714 |                  0.4166 |
| NFCorpus  |                      0.2914 |                    0.2688 |                         0.3009 |                    0.3062 |     0.3225 |                  0.2831 |
| SCIDOCS   |                      0.2018 |                    0.1832 |                         0.2024 |                    0.2058 |     0.2182 |                  0.1955 |
| SciFact   |                      0.594  |                    0.5341 |                         0.6081 |                    0.6108 |     0.6264 |                  0.5672 |
| TRECCOVID |                      0.5198 |                    0.4833 |                         0.506  |                    0.5074 |     0.5082 |                  0.466  |
| Total     |                      0.4106 |                    0.3749 |                         0.4128 |                    0.4167 |     0.4293 |                  0.3857 |


DIM=384//2=192
Without Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.3998 |                    0.3485 |                         0.4128 |                    0.4243 |
| NFCorpus  |                      0.2601 |                    0.2297 |                         0.267  |                    0.2746 |
| SCIDOCS   |                      0.1768 |                    0.1549 |                         0.1904 |                    0.1953 |
| SciFact   |                      0.5266 |                    0.4748 |                         0.5516 |                    0.561  |
| TRECCOVID |                      0.4673 |                    0.3948 |                         0.4809 |                    0.4863 |
| Total     |                      0.3661 |                    0.3205 |                         0.3805 |                    0.3883 |

DIM=384//2=192
With Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.4006 |                    0.3474 |                         0.4275 |                    0.4352 |
| NFCorpus  |                      0.2581 |                    0.2404 |                         0.2754 |                    0.2895 |
| SCIDOCS   |                      0.1829 |                    0.1545 |                         0.1845 |                    0.1911 |
| SciFact   |                      0.5359 |                    0.4763 |                         0.5807 |                    0.5874 |
| TRECCOVID |                      0.4588 |                    0.4396 |                         0.5068 |                    0.5108 |
| Total     |                      0.3673 |                    0.3316 |                         0.395  |                    0.4028 |

DIM=384//3=128
Without Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.3622 |                    0.3006 |                         0.3799 |                    0.3944 |
| NFCorpus  |                      0.2355 |                    0.1992 |                         0.2459 |                    0.2552 |
| SCIDOCS   |                      0.1559 |                    0.1281 |                         0.1703 |                    0.1767 |
| SciFact   |                      0.4694 |                    0.3839 |                         0.4842 |                    0.511  |
| TRECCOVID |                      0.426  |                    0.3716 |                         0.4308 |                    0.4528 |
| Total     |                      0.3298 |                    0.2767 |                         0.3422 |                    0.358  |

DIM=384//3=128
With Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.37   |                    0.3088 |                         0.4004 |                    0.4088 |
| NFCorpus  |                      0.2276 |                    0.2087 |                         0.2619 |                    0.2769 |
| SCIDOCS   |                      0.1626 |                    0.1329 |                         0.1687 |                    0.1732 |
| SciFact   |                      0.4699 |                    0.4084 |                         0.5352 |                    0.5644 |
| TRECCOVID |                      0.4477 |                    0.3719 |                         0.4717 |                    0.493  |
| Total     |                      0.3356 |                    0.2861 |                         0.3676 |                    0.3833 |


DIM=384//4=96
Without Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.3299 |                    0.2619 |                         0.3547 |                    0.3718 |
| NFCorpus  |                      0.2066 |                    0.1666 |                         0.2264 |                    0.2311 |
| SCIDOCS   |                      0.1391 |                    0.1044 |                         0.1545 |                    0.159  |
| SciFact   |                      0.4255 |                    0.3406 |                         0.4515 |                    0.4715 |
| TRECCOVID |                      0.4076 |                    0.3444 |                         0.41   |                    0.4188 |
| Total     |                      0.3017 |                    0.2436 |                         0.3194 |                    0.3304 |

DIM=384//4=96
With Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.3336 |                    0.2747 |                         0.3816 |                    0.3914 |
| NFCorpus  |                      0.2107 |                    0.1684 |                         0.2361 |                    0.2451 |
| SCIDOCS   |                      0.1463 |                    0.1176 |                         0.152  |                    0.1594 |
| SciFact   |                      0.3933 |                    0.3474 |                         0.4749 |                    0.5076 |
| TRECCOVID |                      0.3938 |                    0.3477 |                         0.422  |                    0.4214 |
| Total     |                      0.2955 |                    0.2512 |                         0.3333 |                    0.345  |


DIM=384//8=48
Without Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.2295 |                    0.1477 |                         0.2615 |                    0.2841 |
| NFCorpus  |                      0.1324 |                    0.1044 |                         0.1554 |                    0.1581 |
| SCIDOCS   |                      0.0777 |                    0.0487 |                         0.0918 |                    0.1007 |
| SciFact   |                      0.2803 |                    0.2051 |                         0.3244 |                    0.352  |
| TRECCOVID |                      0.2967 |                    0.214  |                         0.3422 |                    0.3239 |
| Total     |                      0.2033 |                    0.144  |                         0.2351 |                    0.2438 |

DIM=384//8=48
With Matryoshka:
Main Score Results:
| Task      |   Matryoshka_1_5bit_Trained |   Matryoshka_1bit_Trained |   Matryoshka_2bit_3bit_Trained |   Matryoshka_2bit_Trained |
|:----------|----------------------------:|--------------------------:|-------------------------------:|--------------------------:|
| ArguAna   |                      0.241  |                    0.1649 |                         0.3056 |                    0.3194 |
| NFCorpus  |                      0.1396 |                    0.1053 |                         0.1552 |                    0.1708 |
| SCIDOCS   |                      0.0877 |                    0.0641 |                         0.1066 |                    0.1162 |
| SciFact   |                      0.2663 |                    0.202  |                         0.3306 |                    0.3436 |
| TRECCOVID |                      0.2771 |                    0.1654 |                         0.2885 |                    0.3047 |
| Total     |                      0.2023 |                    0.1403 |                         0.2373 |                    0.2509 |