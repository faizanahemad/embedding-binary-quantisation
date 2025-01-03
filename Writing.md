Plots
- Compute Normalized Performance plot for inference retrieval across CPU and GPU devices
- Storage Normalized Performance plot for inference retrieval
- Dimensions Normalized Performance plot for inference retrieval
- Perf plot for 0.5, 1, 1.5, 2 bits as dimension increases with and without Matryoshka training.
- Plots with actual wall clock time and memory usage and storage and FLOPS.

Others
- Mutual Information for how much information is lost in quantization
- Mutual information after Matryoshka training.
- Show that there is a spectrum of performance between full precision and full storage to quantised storage where we can play with number of bits and number of dimensions.
- Need to compare against other ways of quantizing embeddings. 
- Need to compare against other ways of Pruning embedding dimensions.
- QA2 - Quantization Aware Adaptation
- QAMA - Quantization Aware Matryoshka Adaptation



FAQ
- What is the difference between MRL and other methods?
- Why general sparsity based methods are not good? Because all dimensions of embedding are important. Reducing precision works better than sparsity.


Criticisms
- Intro - why do embeddings have high computational cost?