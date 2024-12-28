import numpy as np  
import time  
import matplotlib.pyplot as plt  
from numba import njit  
import seaborn as sns  
  
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
    similarities = dot_products / query_norm  
    return similarities  
  
def optimized_cosine_similarity(query_vector, passage_vectors_norm):  
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
    # Compute dot products (cosine similarities)  
    similarities = passage_vectors_norm @ query_vector_norm  
    return similarities  
  
@njit(parallel=True, fastmath=True, cache=True)
def cosine_similarity_jit(query_vector, passage_vectors):
    """
    Optimized JIT-compiled cosine similarity using vectorized operations.
    
    Args:
        query_vector (np.ndarray): Shape (D,)
        passage_vectors (np.ndarray): Shape (N, D)
    
    Returns:
        similarities (np.ndarray): Shape (N,)
    """
    # Compute query norm once
    query_norm = np.linalg.norm(query_vector)
    
    # Vectorized dot products
    dot_products = passage_vectors @ query_vector
    
    # Vectorized norms computation
    passage_norms = np.sqrt(np.sum(passage_vectors * passage_vectors, axis=1))
    
    # Vectorized division
    similarities = dot_products / (passage_norms * query_norm)
    
    return similarities

@njit(parallel=True, fastmath=True, cache=True)
def cosine_similarity_jit_prenorm(query_vector, passage_vectors_norm):
    """
    Optimized JIT-compiled cosine similarity using vectorized operations.
    
    Args:
        query_vector (np.ndarray): Shape (D,)
        passage_vectors_norm (np.ndarray): Shape (N, D)
    
    Returns:
        similarities (np.ndarray): Shape (N,)
    """
    # Compute dot products (cosine similarities)
    query_norm = np.linalg.norm(query_vector)
    similarities = passage_vectors_norm @ query_vector / query_norm
    return similarities

# Pre-warm the function
def prewarm_similarity_function(N_list, D_list):
    """
    Pre-warm the JIT-compiled function for all N,D combinations.
    
    Args:
        N_list (list): List of N values to pre-warm for
        D_list (list): List of D values to pre-warm for
    """
    print("Pre-warming JIT functions for all sizes...")
    
    for D in D_list:
        for N in N_list:
            # Generate dummy data for this size
            dummy_query = np.random.randn(D).astype(np.float32)
            dummy_passages = np.random.randn(N, D).astype(np.float32)
            dummy_passages_norm = dummy_passages / np.linalg.norm(dummy_passages, axis=1)[:, np.newaxis]
            
            # Pre-warm both JIT functions
            _ = cosine_similarity_jit(dummy_query, dummy_passages)
            _ = cosine_similarity_jit_prenorm(dummy_query, dummy_passages_norm)
            
            print(f"Pre-warmed for N={N}, D={D}")
    
    print("JIT compilation completed for all sizes")
  
def pack_bits(binary_vectors):  
    """  
    Packs binary vectors into bytes.  
  
    Args:  
        binary_vectors (np.ndarray): Binary vectors of shape (N, D), values are 0 or 1.  
  
    Returns:  
        packed_vectors (np.ndarray): Packed vectors of shape (N, num_bytes).  
    """  
    N, D = binary_vectors.shape  
    num_bits = D  
    num_bytes = (num_bits + 7) // 8  
    # Pad the binary vectors to make the number of bits a multiple of 8  
    padding_bits = num_bytes * 8 - num_bits  
    if padding_bits > 0:  
        padded_binary_vectors = np.hstack(  
            [binary_vectors, np.zeros((N, padding_bits), dtype=np.uint8)]  
        )  
    else:  
        padded_binary_vectors = binary_vectors  
    # Pack bits into bytes  
    packed_vectors = np.packbits(padded_binary_vectors, axis=1)  
    return packed_vectors  
  
def hamming_similarity(query_vector_binary, passage_vectors_binary):  
    """  
    Computes Hamming similarity without bit packing.  
  
    Args:  
        query_vector_binary (np.ndarray): Binary vector of shape (D,).  
        passage_vectors_binary (np.ndarray): Binary vectors of shape (N, D).  
  
    Returns:  
        similarities (np.ndarray): Hamming similarities of shape (N,).  
    """  
    # Compute Hamming distance  
    distances = np.count_nonzero(passage_vectors_binary != query_vector_binary, axis=1)  
    # Convert to similarity  
    similarities = 1 - distances / passage_vectors_binary.shape[1]  
    return similarities  
  
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
    # Compute Hamming distances using numpy.bitwise_count  
    distances = np.bitwise_count(xor_result).sum(axis=1)  
    # Convert to similarity  
    similarities = 1 - distances / D  
    return similarities  
  
# Modified function using numpy.bitwise_count  
def compute_hamming_distance_packed(a_packed, b_packed):  
    """  
    Computes the Hamming distance between two packed binary vectors using numpy.bitwise_count.  
  
    Args:  
        a_packed (np.ndarray): Packed binary vector (1D array of uint8).  
        b_packed (np.ndarray): Packed binary vector (1D array of uint8).  
  
    Returns:  
        int: The Hamming distance between the two vectors.  
    """  
    # Compute bitwise XOR to identify differing bits  
    xor_result = np.bitwise_xor(a_packed, b_packed)  
    # Count the number of set bits (1s) in the XOR result  
    hamming_distance = np.bitwise_count(xor_result).sum()  
    return int(hamming_distance)  
  
# Benchmarking Functions  
def benchmark_similarity_methods(N_list, D_list, num_runs=5):
    """
    Benchmarks different similarity computation methods.

    Args:
        N_list (list): List of numbers of passage vectors to test.
        D_list (list): List of dimensions to test.
        num_runs (int): Number of times to run each benchmark for averaging.

    Returns:
        results (dict): Timing results for each method.
    """
    # Pre-warm JIT functions
    t0 = time.perf_counter()
    prewarm_similarity_function(N_list, D_list)
    prewarm_time = time.perf_counter() - t0
    print(f"Pre-warming time: {prewarm_time:.4f} seconds")

    results = {
        'cosine': [], 'cosine_pre_norm': [], 'cosine_optimized': [],
        'cosine_jit': [], 'cosine_jit_prenorm': [],
        'hamming': [], 'hamming_packed': []
    }

    for D in D_list:
        for N in N_list:
            print(f"\nBenchmarking with N={N}, D={D}")
            
            # Generate vectors and prepare data (outside timing)
            query_vector, passage_vectors = generate_vectors(N, D)
            passage_norms = np.linalg.norm(passage_vectors, axis=1)
            passage_vectors_norm = passage_vectors / passage_norms[:, np.newaxis]
            
            # Prepare binary vectors
            threshold = 0.5
            query_vector_binary = (query_vector >= threshold).astype(np.uint8)
            passage_vectors_binary = (passage_vectors >= threshold).astype(np.uint8)
            query_vector_packed = pack_bits(query_vector_binary.reshape(1, -1))[0]
            passage_vectors_packed = pack_bits(passage_vectors_binary)
            
            # Dictionary to store times for this N,D combination
            times = {method: [] for method in results.keys()}
            
            # Warmup run (results discarded)
            for method in results.keys():
                if method == 'cosine':
                    _ = cosine_similarity(query_vector, passage_vectors)
                elif method == 'cosine_pre_norm':
                    _ = cosine_similarity_pre_norm(query_vector, passage_vectors_norm, passage_norms)
                elif method == 'cosine_jit':
                    _ = cosine_similarity_jit(query_vector, passage_vectors)
                elif method == 'cosine_jit_prenorm':
                    _ = cosine_similarity_jit_prenorm(query_vector, passage_vectors_norm)
                elif method == 'hamming':
                    _ = hamming_similarity(query_vector_binary, passage_vectors_binary)
                elif method == 'hamming_packed':
                    _ = hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D)
                    
            # Actual benchmark runs
            for run in range(num_runs):
                for method in results.keys():
                    start_time = time.perf_counter()  # More precise than time.time()
                    
                    if method == 'cosine':
                        _ = cosine_similarity(query_vector, passage_vectors)
                    elif method == 'cosine_pre_norm':
                        _ = cosine_similarity_pre_norm(query_vector, passage_vectors_norm, passage_norms)
                    elif method == 'cosine_optimized':
                        _ = optimized_cosine_similarity(query_vector, passage_vectors_norm)
                    elif method == 'cosine_jit':
                        _ = cosine_similarity_jit(query_vector, passage_vectors)
                    elif method == 'cosine_jit_prenorm':
                        _ = cosine_similarity_jit_prenorm(query_vector, passage_vectors_norm)
                    elif method == 'hamming':
                        _ = hamming_similarity(query_vector_binary, passage_vectors_binary)
                    elif method == 'hamming_packed':
                        _ = hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D)
                    
                    elapsed_time = time.perf_counter() - start_time
                    times[method].append(elapsed_time)
            
            # Store average times
            for method in results.keys():
                avg_time = np.mean(times[method])
                std_time = np.std(times[method])
                print(f"{method}: {avg_time:.4f}s ± {std_time:.4f}s")
                results[method].append((N, D, avg_time, std_time))

    return results

def plot_results(results, N_list, D_list):  
    """  
    Plots the benchmarking results.  
  
    Args:  
        results (dict): Timing results from benchmark_similarity_methods.  
        N_list (list): List of numbers of passage vectors tested.  
        D_list (list): List of dimensions tested.  
    """  
    sns.set(style='whitegrid')  
    methods = ['cosine', 'cosine_pre_norm', 'cosine_optimized', 'cosine_jit', 'cosine_jit_prenorm', 'hamming', 'hamming_packed']  # 'cosine_jit'
  
    # Plot Computation Time vs. Number of Passage Vectors for each D  
    for D in D_list:  
        plt.figure(figsize=(12, 6))  
        for method in methods:  
            times = [t for (N, d, t) in results[method] if d == D]  
            Ns = [N for (N, d, t) in results[method] if d == D]  
            plt.plot(Ns, times, marker='o', label=method)  
        plt.title(f'Computation Time vs. Number of Passage Vectors (D = {D})')  
        plt.xlabel('Number of Passage Vectors (N)')  
        plt.ylabel('Computation Time (s)')  
        plt.xscale('log')  
        plt.legend()  
        plt.show()  
  
    # Plot Computation Time vs. Dimension Size for each N  
    for N in N_list:  
        plt.figure(figsize=(12, 6))  
        for method in methods:  
            times = [t for (n, D, t) in results[method] if n == N]  
            Ds = [D for (n, D, t) in results[method] if n == N]  
            plt.plot(Ds, times, marker='o', label=method)  
        plt.title(f'Computation Time vs. Dimension Size (N = {N})')  
        plt.xlabel('Dimension Size (D)')  
        plt.ylabel('Computation Time (s)')  
        plt.xscale('log')  
        plt.legend()  
        plt.show()  
  
# Main Execution  
if __name__ == "__main__":  
    # Define ranges for N (number of passage vectors) and D (dimensions)  
    N_list = [1000, 5000, 10000, 50000, 100000]  # From 1K to 100K  
    D_list = [256, 512, 1024, 2048, 4096]  # From 256 to 4K  
  
    # Run benchmarks  
    results = benchmark_similarity_methods(N_list, D_list)  
  
    # Plot the results  
    plot_results(results, N_list, D_list)  
