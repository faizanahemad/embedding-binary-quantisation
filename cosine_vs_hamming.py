import numpy as np  
import time  
import matplotlib.pyplot as plt  
from numba import njit  
  
# Set random seed for reproducibility  
np.random.seed(42)  
  
def generate_vectors(N, D):  
    """  
    Generates a set of random vectors.  
  
    Args:  
        N (int): Number of passage vectors.  
        D (int): Dimension of each vector.  
  
    Returns:  
        query_vector (np.ndarray): A 1D array of shape (D,).  
        passage_vectors (np.ndarray): A 2D array of shape (N, D).  
    """  
    query_vector = np.random.rand(D).astype(np.float32)  
    passage_vectors = np.random.rand(N, D).astype(np.float32)  
    return query_vector, passage_vectors  
  
def cosine_similarity(query_vector, passage_vectors):  
    """  
    Computes cosine similarity between the query vector and each passage vector.  
  
    Args:  
        query_vector (np.ndarray): Shape (D,).  
        passage_vectors (np.ndarray): Shape (N, D).  
  
    Returns:  
        similarities (np.ndarray): Cosine similarities of shape (N,).  
    """  
    # Compute dot product  
    dot_products = passage_vectors @ query_vector  
    # Compute norms  
    query_norm = np.linalg.norm(query_vector)  
    passage_norms = np.linalg.norm(passage_vectors, axis=1)  
    # Compute cosine similarities  
    similarities = dot_products / (passage_norms * query_norm)  
    return similarities  
  
def cosine_similarity_pre_norm(query_vector, passage_vectors, passage_norms):  
    """  
    Computes cosine similarity with pre-normalized passage vectors.  
  
    Args:  
        query_vector (np.ndarray): Shape (D,).  
        passage_vectors (np.ndarray): Shape (N, D). Should be pre-normalized.  
        passage_norms (np.ndarray): Pre-computed norms of passage vectors.  
      
    Returns:  
        similarities (np.ndarray): Cosine similarities of shape (N,).  
    """  
    query_norm = np.linalg.norm(query_vector)  
    dot_products = passage_vectors @ query_vector  
    similarities = dot_products / (passage_norms * query_norm)  
    return similarities  
  
def optimized_cosine_similarity(query_vector, passage_vectors):  
    """  
    Optimized cosine similarity computation using vectorization.  
  
    Args:  
        query_vector (np.ndarray): Shape (D,).  
        passage_vectors (np.ndarray): Shape (N, D).  
  
    Returns:  
        similarities (np.ndarray): Cosine similarities of shape (N,).  
    """  
    # Normalize query and passage vectors  
    query_vector_norm = query_vector / np.linalg.norm(query_vector)  
    passage_vectors_norm = passage_vectors / np.linalg.norm(passage_vectors, axis=1, keepdims=True)  
    # Compute dot products (cosine similarities)  
    similarities = passage_vectors_norm @ query_vector_norm  
    return similarities  
  
@njit(parallel=True)  
def cosine_similarity_jit(query_vector, passage_vectors):  
    """  
    JIT-compiled cosine similarity using Numba.  
  
    Args:  
        query_vector (np.ndarray): Shape (D,).  
        passage_vectors (np.ndarray): Shape (N, D).  
  
    Returns:  
        similarities (np.ndarray): Cosine similarities of shape (N,).  
    """  
    N = passage_vectors.shape[0]  
    similarities = np.zeros(N, dtype=np.float32)  
    query_norm = np.linalg.norm(query_vector)  
    for i in range(N):  
        dot_product = np.dot(passage_vectors[i], query_vector)  
        passage_norm = np.linalg.norm(passage_vectors[i])  
        similarities[i] = dot_product / (passage_norm * query_norm)  
    return similarities  
  
def quantize_vectors(vecs, bits):  
    """  
    Quantizes vectors to binary representation.  
  
    Args:  
        vecs (np.ndarray): Vectors to quantize, shape (N, D).  
        bits (int): Number of bits for quantization.  
  
    Returns:  
        binary_vecs (np.ndarray): Quantized binary vectors, shape (N, D * bits).  
    """  
    # Simple threshold quantization (for demonstration)  
    thresholds = np.percentile(vecs, np.linspace(0, 100, 2 ** bits + 1)[1:-1], axis=1)  
    binary_vecs = np.zeros_like(vecs, dtype=np.uint8)  
    for i in range(vecs.shape[0]):  
        for b in range(bits):  
            binary_vecs[i] |= (vecs[i] >= thresholds[i][b]) << b  
    return binary_vecs  
  
def hamming_similarity(query_vector, passage_vectors):  
    """  
    Computes Hamming similarity without bit packing.  
  
    Args:  
        query_vector (np.ndarray): Binary vector of shape (D,).  
        passage_vectors (np.ndarray): Binary vectors of shape (N, D).  
  
    Returns:  
        similarities (np.ndarray): Hamming similarities of shape (N,).  
    """  
    # Compute Hamming distance  
    distances = np.count_nonzero(passage_vectors != query_vector, axis=1)  
    # Convert to similarity  
    similarities = 1 - distances / passage_vectors.shape[1]  
    return similarities  
  
def pack_bits(binary_vectors):  
    """  
    Packs binary vectors into bytes.  
  
    Args:  
        binary_vectors (np.ndarray): Binary vectors of shape (N, D), values are 0 or 1.  
  
    Returns:  
        packed_vectors (np.ndarray): Packed vectors of shape (N, num_bytes).  
    """  
    N, D = binary_vectors.shape  
    num_bytes = (D + 7) // 8  
    padded_bits = np.zeros((N, num_bytes * 8), dtype=np.uint8)  
    padded_bits[:, :D] = binary_vectors  
    packed_vectors = np.packbits(padded_bits, axis=1)  
    return packed_vectors  
  
def hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D):  
    """  
    Computes Hamming similarity using packed binary vectors.  
  
    Args:  
        query_vector_packed (np.ndarray): Packed query vector, shape (num_bytes,).  
        passage_vectors_packed (np.ndarray): Packed passage vectors, shape (N, num_bytes).  
        D (int): Original dimension of vectors.  
  
    Returns:  
        similarities (np.ndarray): Hamming similarities of shape (N,).  
    """  
    # Compute XOR  
    xor_result = np.bitwise_xor(passage_vectors_packed, query_vector_packed)  
    # Compute Hamming distances  
    distances = np.unpackbits(xor_result, axis=1).sum(axis=1)  
    # Limit distances to original dimension in case of padding  
    distances = distances[:passage_vectors_packed.shape[0]]  
    # Convert to similarity  
    similarities = 1 - distances / D  
    return similarities  
  
# Benchmarking Functions  
  
def benchmark_similarity_methods(N_list, D_list):  
    """  
    Benchmarks different similarity computation methods.  
  
    Args:  
        N_list (list): List of numbers of passage vectors to test.  
        D_list (list): List of dimensions to test.  
  
    Returns:  
        results (dict): Timing results for each method.  
    """  
    results = {  
        'cosine': [],  
        'cosine_pre_norm': [],  
        'cosine_optimized': [],  
        'cosine_jit': [],  
        'hamming': [],  
        'hamming_packed': []  
    }  
  
    for D in D_list:  
        for N in N_list:  
            print(f"\nBenchmarking with N={N}, D={D}")  
            # Generate vectors (not included in timing)  
            query_vector, passage_vectors = generate_vectors(N, D)  
            # Pre-normalize passage vectors for cosine_pre_norm  
            passage_norms = np.linalg.norm(passage_vectors, axis=1)  
            # Quantize vectors for Hamming similarity  
            # For simplicity, we will binarize the vectors using a threshold  
            threshold = 0.5  
            query_vector_binary = (query_vector >= threshold).astype(np.uint8)  
            passage_vectors_binary = (passage_vectors >= threshold).astype(np.uint8)  
            # Pack binary vectors  
            query_vector_packed = pack_bits(query_vector_binary.reshape(1, -1))[0]  
            passage_vectors_packed = pack_bits(passage_vectors_binary)  
            # Ensure D for Hamming is original dimension before padding  
            D_hamming = D  
  
            # Time standard cosine similarity  
            start_time = time.time()  
            cosine_similarity(query_vector, passage_vectors)  
            elapsed_time = time.time() - start_time  
            results['cosine'].append((N, D, elapsed_time))  
  
            # Time cosine similarity with pre-normalization  
            start_time = time.time()  
            cosine_similarity_pre_norm(query_vector, passage_vectors, passage_norms)  
            elapsed_time = time.time() - start_time  
            results['cosine_pre_norm'].append((N, D, elapsed_time))  
  
            # Time optimized cosine similarity  
            start_time = time.time()  
            optimized_cosine_similarity(query_vector, passage_vectors)  
            elapsed_time = time.time() - start_time  
            results['cosine_optimized'].append((N, D, elapsed_time))  
  
            # Time JIT-compiled cosine similarity  
            start_time = time.time()  
            cosine_similarity_jit(query_vector, passage_vectors)  
            elapsed_time = time.time() - start_time  
            results['cosine_jit'].append((N, D, elapsed_time))  
  
            # Time Hamming similarity without packing  
            start_time = time.time()  
            hamming_similarity(query_vector_binary, passage_vectors_binary)  
            elapsed_time = time.time() - start_time  
            results['hamming'].append((N, D, elapsed_time))  
  
            # Time Hamming similarity with packing  
            start_time = time.time()  
            hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D_hamming)  
            elapsed_time = time.time() - start_time  
            results['hamming_packed'].append((N, D, elapsed_time))  
  
    return results  
  
def plot_results(results, N_list, D_list):  
    """  
    Plots the benchmarking results.  
  
    Args:  
        results (dict): Timing results from benchmark_similarity_methods.  
        N_list (list): List of numbers of passage vectors tested.  
        D_list (list): List of dimensions tested.  
    """  
    import matplotlib.pyplot as plt  
    import seaborn as sns  
    sns.set(style='whitegrid')  
  
    methods = ['cosine', 'cosine_pre_norm', 'cosine_optimized', 'hamming', 'hamming_packed']  # 'cosine_jit'
    for D in D_list:  
        plt.figure(figsize=(12, 6))  
        for method in methods:  
            times = [t for (N, d, t) in results[method] if d == D]  
            Ns = [N for (N, d, t) in results[method] if d == D]  
            plt.plot(Ns, times, marker='o', label=method)  
        plt.title(f'Computation Time vs. Number of Passage Vectors (D = {D})')  
        plt.xlabel('Number of Passage Vectors (N)')  
        plt.ylabel('Computation Time (s)')  
        plt.legend()  
        plt.show()  
  
    for N in N_list:  
        plt.figure(figsize=(12, 6))  
        for method in methods:  
            times = [t for (n, D, t) in results[method] if n == N]  
            Ds = [D for (n, D, t) in results[method] if n == N]  
            plt.plot(Ds, times, marker='o', label=method)  
        plt.title(f'Computation Time vs. Dimension Size (N = {N})')  
        plt.xlabel('Dimension Size (D)')  
        plt.ylabel('Computation Time (s)')  
        plt.legend()  
        plt.show()  
  
# Main Execution  
  
if __name__ == "__main__":  
    # Define ranges for N (number of passage vectors) and D (dimensions)  
    N_list = [1000, 5000, 10000]   # You can expand this list as needed  
    D_list = [128, 256, 512, 1024] # You can expand this list as needed  
  
    # Run benchmarks  
    results = benchmark_similarity_methods(N_list, D_list)  
  
    # Plot the results  
    plot_results(results, N_list, D_list)  
