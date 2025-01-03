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
    
def pack_bits_optimized(binary_vectors):  
    """  
    Pack binary vectors into uint32 words for better performance.  
      
    Args:  
        binary_vectors (np.ndarray): Binary vectors (N, D).  
      
    Returns:  
        packed_vectors (np.ndarray): Packed vectors (N, ceil(D/32)).  
    """  
    N, D = binary_vectors.shape  
    num_words = (D + 31) // 32  # Ceiling division by 32  
      
    packed = np.zeros((N, num_words), dtype=np.uint32)  
      
    # Vectorized packing  
    for i in range(min(32, D)):  
        packed |= binary_vectors[:, i:i+D:32].astype(np.uint32) << i  
      
    return packed  
  
def pack_bits(binary_vectors, word_size=64):
    """
    Packs binary vectors into uint32 or uint64 words.

    Args:
        binary_vectors (np.ndarray): Binary vectors of shape (N, D), values are 0 or 1.
        word_size (int): Size of words to pack into (32 or 64 bits).

    Returns:
        packed_vectors (np.ndarray): Packed vectors of shape (N, num_words).
    """
    if word_size not in [32, 64]:
        raise ValueError("word_size must be either 32 or 64")

    N, D = binary_vectors.shape
    dtype = np.uint32 if word_size == 32 else np.uint64
    
    # Calculate number of words needed
    num_words = (D + word_size - 1) // word_size
    padding_bits = num_words * word_size - D
    
    # Pad the vectors if necessary
    if padding_bits > 0:
        padded_vectors = np.hstack([
            binary_vectors,
            np.zeros((N, padding_bits), dtype=np.uint8)
        ])
    else:
        padded_vectors = binary_vectors

    # Reshape to prepare for packing
    bits_reshaped = padded_vectors.reshape(N, -1, word_size)
    
    # Pack bits using vectorized operations
    packed = np.zeros((N, num_words), dtype=dtype)
    for i in range(word_size):
        packed |= bits_reshaped[:, :, i].astype(dtype) << (word_size - 1 - i)
    
    return packed

def unpack_bits(packed_vectors, original_dim, word_size=64):
    """
    Unpacks uint32 or uint64 words back into binary vectors.

    Args:
        packed_vectors (np.ndarray): Packed vectors of shape (N, num_words).
        original_dim (int): Original dimension of binary vectors.
        word_size (int): Size of words that were packed (32 or 64 bits).

    Returns:
        binary_vectors (np.ndarray): Unpacked binary vectors of shape (N, original_dim).
    """
    if word_size not in [32, 64]:
        raise ValueError("word_size must be either 32 or 64")

    N = packed_vectors.shape[0]
    dtype = np.uint32 if word_size == 32 else np.uint64
    
    # Create output array
    unpacked = np.zeros((N, packed_vectors.shape[1] * word_size), dtype=np.uint8)
    
    # Unpack bits using vectorized operations
    for i in range(word_size):
        shift = word_size - 1 - i
        unpacked[:, i::word_size] = ((packed_vectors & (dtype(1) << shift)) >> shift).astype(np.uint8)
    
    # Return only the original dimensions
    return unpacked[:, :original_dim]
  
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
    distances = np.bitwise_count(xor_result)
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

def quantize_to_n_bits(query_vector, passage_vectors, bits=2):  
    """  
    Quantizes vectors to n bits per dimension, handling non-integer bits.  
  
    Args:  
        query_vector (np.ndarray): Query vector of shape (D,).  
        passage_vectors (np.ndarray): Passage vectors of shape (N, D).  
        bits (float): Number of bits per dimension (can be non-integer).  
  
    Returns:  
        query_vector_quantized (np.ndarray): Quantized query vector.  
        passage_vectors_quantized (np.ndarray): Quantized passage vectors.  
    """  
    # Calculate the number of levels  
    levels = int(2 ** bits)  
    # Determine quantization bins based on the data distribution  
    min_val = min(np.min(query_vector), np.min(passage_vectors))  
    max_val = max(np.max(query_vector), np.max(passage_vectors))  
    bins = np.linspace(min_val, max_val, levels + 1)[1:-1]  # Exclude first and last for bin edges  
  
    # Quantize query vector  
    query_vector_quantized = np.digitize(query_vector, bins)  
    # Quantize passage vectors  
    passage_vectors_quantized = np.digitize(passage_vectors, bins)  
  
    return query_vector_quantized, passage_vectors_quantized  
 
  
def expand_vectors(query_vector_quantized, passage_vectors_quantized, bits=2):
    """  
    Expands quantized vectors into binary codewords using a codebook.
    Optimized version using vectorized operations.
    
    Args:  
        query_vector_quantized (np.ndarray): Quantized query vector.  
        passage_vectors_quantized (np.ndarray): Quantized passage vectors.  
        bits (float): Number of bits per dimension.  
    
    Returns:  
        query_vector_expanded (np.ndarray): Expanded binary query vector.  
        passage_vectors_expanded (np.ndarray): Expanded binary passage vectors.  
    """
    # Create lookup tables as numpy arrays
    if bits == 1:
        lookup = np.array([[0], [1]], dtype=np.uint8)
    elif bits == 1.7:
        lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.uint8)
    elif bits == 2:
        lookup = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    elif bits == 3:
        lookup = np.array([
            [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]
        ], dtype=np.uint8)
    else:
        raise ValueError("Unsupported number of bits for codebook mapping.")

    # Vectorized expansion for query vector
    query_vector_expanded = lookup[query_vector_quantized.ravel()].ravel()
    
    # Vectorized expansion for passage vectors
    N, D = passage_vectors_quantized.shape
    codeword_length = lookup.shape[1]
    passage_vectors_expanded = lookup[passage_vectors_quantized.ravel()].reshape(N, D * codeword_length)

    return query_vector_expanded, passage_vectors_expanded

  
# Benchmarking Functions  
def benchmark_similarity_methods(N_list, D_list, num_runs=5):  
    """  
    Benchmarks different similarity computation methods, including the 1.7-bit vectors.  
  
    Args:  
        N_list (list): List of numbers of passage vectors to test.  
        D_list (list): List of dimensions to test.  
        num_runs (int): Number of times to run each benchmark for averaging.  
  
    Returns:  
        results (dict): Timing results for each method.  
    """  
    # Pre-warm JIT functions (if applicable)  
    t0 = time.perf_counter()  
    prewarm_similarity_function(N_list, D_list)  
    prewarm_time = time.perf_counter() - t0  
    print(f"Pre-warming time: {prewarm_time:.4f} seconds")  
  
    # Initialize results dictionary  
    results = {  
        'cosine': [], 'cosine_pre_norm': [], 'cosine_optimized': [],  
        # 'cosine_jit': [], 'cosine_jit_prenorm': [],  
        'hamming': [], 'hamming_packed': [],  
        # 'hamming_1_7bit': [], 'hamming_1_7bit_packed': [],  
        'hamming_2bit': [], 'hamming_2bit_packed': [],  
        'hamming_3bit': [], 'hamming_3bit_packed': []  
    }  
  
    # Bits configurations to benchmark  
    bits_list = [1, 1.7, 2, 3]  # Include 1.7 bits  
  
    for D in D_list:  
        for N in N_list:  
            print(f"\nBenchmarking with N={N}, D={D}")  
            
            t_s = time.perf_counter()
  
            # Generate vectors and prepare data (outside timing)  
            query_vector, passage_vectors = generate_vectors(N, D)  
            passage_norms = np.linalg.norm(passage_vectors, axis=1)  
            passage_vectors_norm = passage_vectors / passage_norms[:, np.newaxis]  
  
            # Dictionary to store times for this N,D combination  
            times = {method: [] for method in results.keys()}  
  
            # Prepare data for each bits configuration  
            data_prepared = {}  
            for bits in bits_list:  
                # Quantize vectors  
                q_query_vector, q_passage_vectors = quantize_to_n_bits(query_vector, passage_vectors, bits=bits)  
                # Expand vectors  
                e_query_vector, e_passage_vectors = expand_vectors(q_query_vector, q_passage_vectors, bits=bits)  
                # Pack vectors  
                p_query_vector = pack_bits(e_query_vector.reshape(1, -1))[0]  
                p_passage_vectors = pack_bits(e_passage_vectors)  
                # Store prepared data  
                data_prepared[bits] = {  
                    'query_vector': q_query_vector,
                    'passage_vectors': q_passage_vectors,
                    'expanded_query': e_query_vector,  
                    'expanded_passages': e_passage_vectors,  
                    'packed_query': p_query_vector,  
                    'packed_passages': p_passage_vectors,  
                    'codeword_length': e_query_vector.shape[0] // D  
                }  
                
            t_e = time.perf_counter()
            print(f"Data preparation time: {t_e - t_s:.4f} seconds")
  
            # Warmup run (results discarded)  
            for method in results.keys():  
                if method.startswith('cosine'):  
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
                elif method.startswith('hamming'):  
                    if "7bit" in method or "jit" in method:
                        continue
                    if method == 'hamming':  
                        # 1-bit vectors  
                        threshold = 0.5  
                        query_vector_binary = (query_vector >= threshold).astype(np.uint8)  
                        passage_vectors_binary = (passage_vectors >= threshold).astype(np.uint8)  
                        _ = hamming_similarity(query_vector_binary, passage_vectors_binary)  
                    elif method == 'hamming_packed':  
                        query_vector_packed = pack_bits(query_vector_binary.reshape(1, -1))[0]  
                        passage_vectors_packed = pack_bits(passage_vectors_binary)  
                        _ = hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D)  
                    else:  
                        bits = float(method.split('_')[1].replace('bit', '').replace('_packed', ''))  
                        prepared = data_prepared[bits]  
                        if 'packed' in method:  
                            _ = hamming_similarity_packed(  
                                prepared['packed_query'], prepared['packed_passages'], D * prepared['codeword_length']  
                            )  
                        else:  
                            _ = hamming_similarity(  
                                prepared['expanded_query'], prepared['expanded_passages']  
                            )  
  
            # Actual benchmark runs  
            for run in range(num_runs):  
                for method in results.keys():  
                    start_time = time.perf_counter()  
  
                    if method.startswith('cosine'):  
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
                    elif method.startswith('hamming'): 
                        if "7bit" in method or "jit" in method:
                            continue
                        if method == 'hamming':  
                            # 1-bit vectors  
                            _ = hamming_similarity(query_vector_binary, passage_vectors_binary)  
                        elif method == 'hamming_packed':  
                            _ = hamming_similarity_packed(query_vector_packed, passage_vectors_packed, D)  
                        else:  
                            bits = float(method.split('_')[1].replace('bit', '').replace('_packed', ''))  
                            prepared = data_prepared[bits]  
                            if 'packed' in method:  
                                _ = hamming_similarity_packed(  
                                    prepared['packed_query'], prepared['packed_passages'], D * prepared['codeword_length']  
                                )  
                            else:  
                                _ = hamming_similarity(  
                                    prepared['expanded_query'], prepared['expanded_passages']  
                                )  
  
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
    Plots the benchmarking results, including the 1.7-bit methods.  
  
    Args:  
        results (dict): Timing results from benchmark_similarity_methods.  
        N_list (list): List of numbers of passage vectors tested.  
        D_list (list): List of dimensions tested.  
    """  
    sns.set(style='whitegrid')  
  
    methods = [  
        'cosine', 'cosine_pre_norm', 'cosine_optimized',  
        # 'cosine_jit', 'cosine_jit_prenorm',  
        # 'hamming', 
        'hamming_packed',  
        # 'hamming_1_7bit', 'hamming_1_7bit_packed',  
        # 'hamming_2bit', 
        'hamming_2bit_packed',  
        # 'hamming_3bit', 
        'hamming_3bit_packed'  
    ]  
  
    method_labels = {  
        'cosine': 'Cosine',  
        'cosine_pre_norm': 'Cosine Pre-Norm',  
        'cosine_optimized': 'Cosine Optimized',  
        'cosine_jit': 'Cosine JIT',  
        # 'cosine_jit_prenorm': 'Cosine JIT Pre-Norm',  
        # 'hamming': 'Hamming 1-bit',  
        'hamming_packed': 'Hamming 1-bit Packed',  
        # 'hamming_1_7bit': 'Hamming 1.7-bit',  
        # 'hamming_1_7bit_packed': 'Hamming 1.7-bit Packed',  
        # 'hamming_2bit': 'Hamming 2-bit',  
        'hamming_2bit_packed': 'Hamming 2-bit Packed',  
        # 'hamming_3bit': 'Hamming 3-bit',  
        'hamming_3bit_packed': 'Hamming 3-bit Packed'  
    }  
  
    # Colors and markers for different methods  
    import itertools  
    markers = itertools.cycle(('o', 'v', '^', '<', '>', 's', 'p', '*', 'D', 'X', 'h', '+', 'x'))  
    colors = sns.color_palette('husl', n_colors=len(methods))  
  
    # Plot Computation Time vs. Number of Passage Vectors for each D  
    for D in D_list:  
        plt.figure(figsize=(12, 6))  
        for idx, method in enumerate(methods):  
            times = [t for (N, d, t, _) in results[method] if d == D]  
            Ns = [N for (N, d, t, _) in results[method] if d == D]  
            plt.plot(Ns, times, marker=next(markers), color=colors[idx], label=method_labels[method])  
        plt.title(f'Computation Time vs. Number of Passage Vectors (D = {D})')  
        plt.xlabel('Number of Passage Vectors (N)')  
        plt.ylabel('Computation Time (s)')  
        plt.xscale('log')  
        plt.legend()  
        plt.show()  
  
    # Plot Computation Time vs. Dimension Size for each N  
    for N in N_list:  
        plt.figure(figsize=(12, 6))  
        for idx, method in enumerate(methods):  
            times = [t for (n, D, t, _) in results[method] if n == N]  
            Ds = [D for (n, D, t, _) in results[method] if n == N]  
            plt.plot(Ds, times, marker=next(markers), color=colors[idx], label=method_labels[method])  
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
