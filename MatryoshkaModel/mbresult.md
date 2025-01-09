Modern Bert

Main Score Results:
| Task      |     fp32   |   1 bit (Non Trained for Quant) |
|:----------|-----------:|--------------------------------:|
| ArguAna   |     0.5056 |                          0.472  |
| NFCorpus  |     0.3    |                          0.2655 |
| SCIDOCS   |     0.2001 |                          0.1711 |
| SciFact   |     0.6765 |                          0.6118 |
| TRECCOVID |     0.678  |                          0.6594 |
| Total     |     0.472  |                          0.436  |


 (Dim=256)

Main Score Results: (1 epoch)
| Task      |    fp32  |   fp32 (Matryoshka) |   2 bit |   1.5 bit |   1 bit |
|:----------|---------:|--------------------:|--------:|----------:|--------:|
| ArguAna   |   0.4875 |              0.5104 | 0.4924  |   0.4592  | 0.4253  |
| NFCorpus  |   0.2821 |              0.2952 | 0.2674  |   0.2474  | 0.2061  |
| SCIDOCS   |   0.1844 |              0.1954 | 0.1724  |   0.1600  | 0.1303  |
| SciFact   |   0.6405 |              0.6602 | 0.6416  |   0.6144  | 0.5310  |
| TRECCOVID |   0.6849 |              0.6790 | 0.6427  |   0.6332  | 0.5527  |
| Total     |   0.4559 |              0.4680 | 0.4433  |   0.4228  | 0.3691  |

Threshold Only (Non Matryoshka)
Main Score Results:
| Task      |   2 bit |   1.5 bit |   1 bit |
|:----------|--------:|----------:|--------:|
| ArguAna   |  0.4739 |    0.4473 |  0.4039 |
| NFCorpus  |  0.2624 |    0.2357 |  0.2023 |
| SCIDOCS   |  0.1733 |    0.1563 |  0.1261 |
| SciFact   |  0.5826 |    0.5759 |  0.4655 |
| TRECCOVID |  0.6558 |    0.6673 |  0.5449 |
| Total     |  0.4296 |    0.4165 |  0.3485 |

Dim = 768
Main Score Results:
| Task      |   2 bit |   1.5 bit |   1 bit |
|:----------|--------:|----------:|--------:|
| ArguAna   |  0.4981 |    0.4886 |  0.4782 |
| NFCorpus  |  0.2970 |    0.2881 |  0.2671 |
| SCIDOCS   |  0.1909 |    0.1919 |  0.1756 |
| SciFact   |  0.6623 |    0.6535 |  0.6193 |
| TRECCOVID |  0.6954 |    0.6460 |  0.6503 |
| Total     |  0.4687 |    0.4536 |  0.4381 |

Threshold Only (Non Matryoshka)
Main Score Results:
| Task      |   2 bit |   1.5 bit |   1 bit |
|:----------|--------:|----------:|--------:|
| ArguAna   |  0.5013 |    0.4886 |  0.4677 |
| NFCorpus  |  0.2965 |    0.2912 |  0.2694 |
| SCIDOCS   |  0.1958 |    0.1922 |  0.1764 |
| SciFact   |  0.6628 |    0.6651 |  0.6081 |
| TRECCOVID |  0.6754 |    0.6822 |  0.6633 |
| Total     |  0.4664 |    0.4639 |  0.4370 |



Main Score Results:
| Task                    |     fp32   |   1 bit (Non Trained for Quant) |
|:------------------------|-----------:|--------------------------------:|
| ArguAna                 |     0.5056 |                          0.472  |
| CQADupstackTexRetrieval |     0.2667 |                          0.2293 |
| ClimateFEVER            |     0.2776 |                          0.2102 |
| DBPedia                 |     0.3182 |                          0.2571 |
| FEVER                   |     0.6695 |                          0.5388 |
| FiQA2018                |     0.3900 |                          0.3344 |
| HotpotQA                |     0.5354 |                          0.4151 |
| MSMARCO                 |     0.5898 |                          0.5063 |
| NFCorpus                |     0.3000 |                          0.2655 |
| NQ                      |     0.4769 |                          0.3843 |
| QuoraRetrieval          |     0.8782 |                          0.8648 |
| SCIDOCS                 |     0.2001 |                          0.1711 |
| SciFact                 |     0.6765 |                          0.6118 |
| TRECCOVID               |     0.6780 |                          0.6594 |
| Touche2020              |     0.2140 |                          0.1930 |
| Total                   |     0.4651 |                          0.4075 |