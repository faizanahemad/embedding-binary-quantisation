## What we have done till now.
 We developed quantization modules for embedding methods.
 Initially we quantized using binary quantization with trainable scale and threshold per dimension of embedding.

 Losses used for training were:
 - Similarity loss
 - Rank preservation loss
 - KL divergence between original and quantized embedding
 - Contrastive loss with pairwise shuffle.

 Losses used for quantization were:
 - Quantization loss

 Losses to promote Matryoshka 
 - information bottleneck loss (L1 loss on the embedding dimension which increases with the number of dimensions for later dimensions)
 - Orthogonality Regularization

Other characteristics of loss:
- Multi-Scale Loss Function


Qaunt types
- 0.5 bit
- 1 bit
- 1.5 bit (2 thresholds, expands to 2 bit)
- 2 bit (3 thresholds, expands to 3 bit)

# What didn't work?
- Dimension pruning based on Mutual Information or high variance dimension choice or grad of embeddings wrt to loss.
- Smooth training of threshold using sigmoid vs hard threshold using percentiles. Percentile based thresholding is better and faster and more stable especially for larger number of thresholds. Mathematically both are same but trainable thresholds are dependent on initializations and training dynamics. Sigmoid even with temperature annealing is not stable. We used sigmoid with STE straight through estimator.



# Hamming distance and 2 bit to 3 bit expansion

**Understanding the Problem**

- **Embeddings Representation**: We have embeddings where each dimension is quantized into one of four possible values (0, 1, 2, or 3), represented using **2 binary bits per dimension**.
- **Similarity Computation Constraints**: The available operations for computing similarity are:
  - **Euclidean Distance**
  - **Cosine Similarity**
  - **Dot Product Similarity**
  - For **binary embeddings**: **XNOR** followed by **sum**, which effectively counts the number of matching bits (higher sum indicates higher similarity).

**Objective**

- Determine how to compute similarity between these 2-bit-per-dimension embeddings using only the allowed operations.
- Specifically, explore possibilities for:
  - **Using XNOR and sum** to compute similarity.
  - Computing similarity using **cosine similarity**, **dot product**, or **Euclidean distance** without modifications.
- If it's not possible, provide explanations as to why.

---

**Approach**

1. **Assessing XNOR and Sum for 2-Bit Embeddings**
   - Analyze whether it's feasible to compute meaningful similarity using XNOR and sum on embeddings with 2 bits per dimension.
   - Consider possible transformations to fit into the binary embedding framework.

2. **Evaluating Cosine Similarity, Dot Product, and Euclidean Distance**
   - Determine if these operations can be directly applied to the 2-bit quantized embeddings.
   - Explore any necessary adjustments or considerations.

---

### 1. Using XNOR and Sum for 2-Bit Embeddings

**Challenges**

- **Binary Similarity Operations**: XNOR and sum are designed for **binary embeddings**, where each dimension is a single bit (0 or 1). The XNOR operation counts matching bits between two embeddings.
- **2-Bit Representation**: Our embeddings use **2 bits per dimension**, representing integer values from 0 to 3. Directly applying XNOR and sum on these bits may not yield a meaningful similarity measure.

**Detailed Analysis**

- **Direct Application of XNOR and Sum**

  - **Bit-Level Representation**:
    - Each 2-bit value is represented by its binary form:
      - `0` → `00`
      - `1` → `01`
      - `2` → `10`
      - `3` → `11`
    - Concatenating these for all dimensions gives a binary embedding of length `2 * N`, where `N` is the number of dimensions.

  - **Issue with Bitwise Comparison**:
    - The **matching of bits** in the binary representations does not necessarily correspond to the similarity between the original integer values.
    - **Example**:
      - Values `1` (`01`) and `2` (`10`) have no matching bits (`XNOR` gives `00`), yet their integer difference is minimal (|1 - 2| = 1).
      - Values `0` (`00`) and `3` (`11`) have no matching bits (`XNOR` gives `00`), but their integer difference is maximal (|0 - 3| = 3).

- **Resulting Problem**:

  - **Hamming Distance Limitation**:
    - The Hamming distance (number of differing bits) between binary representations does not accurately reflect the **absolute difference** between the integer values.
    - **Conclusion**: Directly using XNOR and sum on the binary representations will not provide a meaningful similarity measure for the embeddings.

**Possible Solutions**

- **Option A: One-Hot Encoding**

  - **Method**:
    - Represent each of the four quantized values using a **one-hot vector** of length 4:
      - `0` → `[1, 0, 0, 0]`
      - `1` → `[0, 1, 0, 0]`
      - `2` → `[0, 0, 1, 0]`
      - `3` → `[0, 0, 0, 1]`
    - Concatenate these vectors for all dimensions, resulting in a binary embedding of length `4 * N`.

  - **Advantages**:
    - Now each dimension is represented using binary bits suitable for XNOR and sum.
    - The similarity computed via XNOR and sum directly counts the number of exact matches between dimensions.

  - **Disadvantages**:
    - **Increases Embedding Size**: Embedding size quadruples from `2 * N` bits to `4 * N` bits, leading to higher storage and computational requirements.

- **Option B: Hamming Distance Mapping**

  - **Method**:
    - Map the 2-bit quantized values to binary codes of length **3 bits** such that the Hamming distance between codes corresponds to the absolute difference between values.
    - **Encoding**:
      - `0` → `000`
      - `1` → `001`
      - `2` → `011`
      - `3` → `111`
    - Hamming distances:
      - Between `0` and `1`: 1 (|0 - 1| = 1)
      - Between `1` and `2`: 1 (|1 - 2| = 1)
      - Between `2` and `3`: 1 (|2 - 3| = 1)
      - Between `0` and `2`: 2 (|0 - 2| = 2)
      - Between `1` and `3`: 2 (|1 - 3| = 2)
      - Between `0` and `3`: 3 (|0 - 3| = 3)

  - **Advantages**:
    - Hamming distance between codes accurately represents the absolute difference between values.
    - Allows meaningful similarity computation using XNOR and sum.

  - **Disadvantages**:
    - **Increases Embedding Size**: Embedding size increases from `2 * N` bits to `3 * N` bits.
    - **Not Optimal**: Still incurs additional storage overhead.

**Conclusion for XNOR and Sum**

- **Not Possible Without Embedding Expansion**:
  - Without increasing the size of the embeddings (which may not be acceptable), it is **not possible** to use XNOR and sum to compute a meaningful similarity between 2-bit embeddings.
  - **Reason**: The binary representations of the 2-bit values do not preserve the necessary properties for Hamming distance to reflect the actual similarity or difference between values.

---


# Expand 2 bit to 3 bit


### 1. Expanding 2-Bit Embeddings to 3 Bits for XNOR and SUM Operations

**Objective**: Expand 2-bit embeddings to 3-bit representations to utilize efficient bitwise operations (XNOR and SUM) for similarity computation, while assessing if cosine similarity remains applicable.

#### **1.1. Rationale for Expanding to 3 Bits**

- **Problem with 2-Bit Representations**:

  - Using XNOR and SUM directly on 2-bit representations does not yield meaningful similarity measures because the binary difference does not correspond to the numerical difference.

- **Solution**:

  - **Design a 3-bit codebook** where the Hamming distance between codes reflects the absolute difference between quantization levels.

  - This mapping ensures that bitwise similarity computations (using XNOR and SUM) correlate with the intended distance between embedding vectors.

#### **1.2. Designing the 3-Bit Codebook**

- **Mapping Quantization Levels to 3-Bit Codes**:

  | Quantization Level (Integer Value) | Binary Representation | 3-Bit Code |
  |------------------------------------|-----------------------|------------|
  | 0                                  | `00`                  | `000`      |
  | 1                                  | `01`                  | `001`      |
  | 2                                  | `10`                  | `011`      |
  | 3                                  | `11`                  | `111`      |

- **Properties of the Codebook**:

  - The **Hamming distance** between the codes corresponds to the **absolute difference** between quantization levels.

  - For example:
    - Hamming distance between `000` and `001` is 1 (difference between levels 0 and 1).
    - Hamming distance between `001` and `011` is 1 (difference between levels 1 and 2).
    - Hamming distance between `000` and `111` is 3 (difference between levels 0 and 3).

#### **1.3. Similarity Computation Using XNOR and SUM**

- **Process**:

  - **Step 1**: Convert each quantized value in the embeddings to its corresponding 3-bit code.

  - **Step 2**: Concatenate these codes to form a binary embedding of length `3 * N`, where `N` is the number of dimensions.

  - **Step 3**: Compute similarity using XNOR and SUM:
    - **XNOR** the embeddings to get a similarity vector.
    - **SUM** the ones in the similarity vector to get the similarity score.

- **Interpretation**:

  - Higher sums indicate more similarity (more bits matching), reflecting smaller distances between quantized levels.

#### **1.4. Applicability of Cosine Similarity**

- **Considerations**:

  - **Binary Embeddings**: After expanding to 3 bits, embeddings are binary vectors.

  - **Cosine Similarity on Binary Vectors**:
    - Cosine similarity can be computed between binary vectors.
    - It measures the cosine of the angle between two vectors, considering their direction and magnitude.

- **Analysis**:

  - **Magnitude Variation**:
    - In binary vectors, the magnitude (vector norm) depends on the number of ones.
    - The variation in magnitude can affect the cosine similarity value.

  - **Information Preservation**:
    - The 3-bit codes are designed to preserve Hamming distances corresponding to quantization level differences.
    - However, cosine similarity might not fully capture this relationship due to its sensitivity to vector lengths.

- **Conclusion**:

  - **Not Ideal**: While cosine similarity can be computed on the expanded embeddings, it might not be the most meaningful similarity measure in this context.

  - **Recommendation**: Using **Hamming distance via XNOR and SUM** is more aligned with the designed codebook and the intended distance measure between embeddings.

### 2. Training to Find Thresholds for 2-Bit Quantization

**Objective**: Develop a method to learn the per-dimension thresholds (three thresholds per dimension) needed to quantize embeddings into four levels (0-3) using your similarity dataset.

#### **2.1. Quantization Process with Multiple Thresholds**

- **Quantization Levels**: For 2-bit quantization, we need to define three thresholds \( t_1, t_2, t_3 \) per dimension to partition the embedding values into four intervals corresponding to quantization levels 0-3.

- **Quantization Function**:

  For each dimension \( d \) and embedding value \( e_d \):

  \[
  q_d = \begin{cases}
    0 & \text{if } e_d \leq t_{1d} \\
    1 & \text{if } t_{1d} < e_d \leq t_{2d} \\
    2 & \text{if } t_{2d} < e_d \leq t_{3d} \\
    3 & \text{if } e_d > t_{3d}
  \end{cases}
  \]

#### **2.2. Learning the Thresholds**

- **Objective**: Optimize the thresholds \( \{ t_{1d}, t_{2d}, t_{3d} \} \) per dimension \( d \) to minimize a loss function that reflects the similarity between embeddings as per your dataset.

- **Approach**:

  - **Define a Differentiable Approximation**: Use a soft quantization function to allow gradient flow during training.

  - **Loss Function**: Design a loss function that penalizes discrepancies between the computed similarities of the quantized embeddings and the ground truth similarities from the dataset.



# Storing binary embeddings
How can I store the quantised embeddings in normal int8 storage in numpy or other formats and then use xnor and sum for hamming distance simply. Essentially how do I store the quantised embeddings after passing binary=True in a effective way to reduce storage while allowing xnor and sum for hamming distance and fast similarity computation for retrieval and ranking using the quantised embeddings


**Introduction**

To efficiently store and compute similarities using quantized embeddings for retrieval and ranking tasks, it's important to represent the binary embeddings in a compact form that allows for fast bitwise operations. When you set `binary=True` in your `QuantizationModule`, the output is likely a tensor containing floating-point values of 0.0 and 1.0. Storing these embeddings as floating-point numbers is inefficient in terms of storage and computational speed.

In this response, I'll explain how to:

1. Convert the binary embeddings from floating-point tensors to packed integers (e.g., `uint8` arrays), significantly reducing storage requirements.
2. Use bitwise operations, specifically XOR (exclusive OR) and popcount (population count), to compute the Hamming distance efficiently for fast similarity computations.

**Converting Binary Embeddings to Packed Integer Representation**

The first step is to convert the binary embeddings from floating-point tensors to a packed bit representation. This involves:

- **Thresholding**: Ensure the binary embeddings are in integer form (0 or 1).
- **Bit Packing**: Combine groups of bits into bytes (or larger integer types), reducing the storage size.

Here’s how you can do it:

1. **Convert Float Tensor to Integer Bits**

   Ensure that your binary embeddings are of type `torch.uint8` or `np.uint8`, containing values 0 and 1.

   ```python
   # Assuming binary_embeddings is a torch.Tensor with values 0.0 and 1.0
   binary_embeddings = binary_embeddings.round().byte()  # Convert to 0 and 1 as uint8
   ```

2. **Reshape and Pad the Bit Array**

   - **Reshape** the bit array so that bits can be grouped into bytes.
   - **Pad** the array if the total number of bits is not a multiple of 8.

   ```python
   num_bits = binary_embeddings.size(1)
   num_padding_bits = (8 - (num_bits % 8)) % 8  # Calculate the required padding

   # Pad with zeros (or ones, depending on your preference)
   padding = torch.zeros(binary_embeddings.size(0), num_padding_bits, dtype=torch.uint8)
   padded_bits = torch.cat((binary_embeddings, padding), dim=1)
   ```

3. **Bit Packing**

   - **Group** bits into bytes.
   - **Pack** them using bitwise operations.

   ```python
   # Reshape to (batch_size, num_bytes, 8)
   bits_reshaped = padded_bits.view(binary_embeddings.size(0), -1, 8)

   # Initialize an empty tensor to store the packed bytes
   packed_embeddings = torch.zeros(bits_reshaped.size(0), bits_reshaped.size(1), dtype=torch.uint8)

   # Iterate over the 8 bits to pack them into a byte
   for i in range(8):
       packed_embeddings |= (bits_reshaped[:, :, i] << (7 - i))
   ```

   Alternatively, you can use more efficient vectorized operations or utilize numpy's `packbits` function if working with numpy arrays.

**Storing Packed Embeddings**

Now, the `packed_embeddings` tensor contains your embeddings in a compact form, where each byte represents 8 bits of your original binary embedding. This reduces the storage requirement by a factor of 8 compared to storing each bit as a separate byte.

To store the embeddings:

- **Convert to Numpy Array (if needed)**:

  ```python
  packed_embeddings_np = packed_embeddings.cpu().numpy()
  ```

- **Save to Disk**:

  You can save the numpy array to disk using numpy's save functions, or use memory-mapped files for large datasets.

  ```python
  # Save to a binary file
  with open('embeddings.bin', 'wb') as f:
      f.write(packed_embeddings_np.tobytes())
  ```

**Computing Hamming Distance Using Bitwise Operations**

To compute the Hamming distance between two packed embeddings, you can use bitwise XOR (to find differing bits) and a population count (to count the number of ones, i.e., differing bits).

Here's how you can do it:

1. **Load the Packed Embeddings**

   ```python
   # Assuming you have two embeddings: embedding_a and embedding_b
   embedding_a = packed_embeddings_np[0]  # Shape: (num_bytes,)
   embedding_b = packed_embeddings_np[1]  # Shape: (num_bytes,)
   ```

2. **Compute Bitwise XOR**

   ```python
   # Perform bitwise XOR to find bits that are different
   xor_result = np.bitwise_xor(embedding_a, embedding_b)
   ```

3. **Compute Hamming Distance (Popcount)**

   - Use a precomputed lookup table for popcount (population count) of bytes.
   - Or use efficient functions available in libraries like NumPy.

   **Using Numpy's Built-in Functions**:

   ```python
   # Numpy approach to compute the number of set bits in the XOR result
   hamming_distance = np.unpackbits(xor_result).sum()
   ```

   **Using a Lookup Table**:

   ```python
   # Precompute a lookup table for the number of set bits in a byte
   popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

   # Use the lookup table to compute the Hamming distance
   hamming_distance = popcount_table[xor_result].sum()
   ```

   This method is efficient because it avoids unpacking bits and operates directly on the packed byte representation.

4. **Vectorized Computation for Multiple Embeddings**

   For retrieval tasks, you often need to compute the Hamming distance between a query embedding and a large number of stored embeddings.

   ```python
   # Let's say query_embedding is of shape (num_bytes,)
   # database_embeddings is of shape (num_embeddings, num_bytes)

   # Compute XOR between the query and all embeddings in the database
   xor_results = np.bitwise_xor(database_embeddings, query_embedding)

   # Compute Hamming distances using the lookup table
   hamming_distances = popcount_table[xor_results].sum(axis=1)
   ```

   This will give you a vector of Hamming distances from the query to each embedding in the database.

**Optimizing for Computational Efficiency**

- **Use Optimized Libraries**: Libraries like NumPy and PyTorch are optimized for such operations and can utilize vectorized computations and underlying hardware acceleration.

- **GPU Acceleration**: If working with PyTorch tensors on a GPU, you can perform these operations on the GPU for further speed improvements.

  ```python
  # Assuming packed_embeddings is a torch.Tensor on GPU
  popcount_lut = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.uint8, device='cuda')

  # Compute XOR
  xor_results = torch.bitwise_xor(database_embeddings, query_embedding.unsqueeze(0))

  # Compute Hamming distances
  hamming_distances = popcount_lut[xor_results].sum(dim=1)
  ```

**Example Code**

Here's a complete example demonstrating the conversion and computation:

```python
import torch
import numpy as np

def pack_bits(binary_embeddings):
    """
    Packs binary embeddings (values 0 and 1) into uint8 format.

    Args:
        binary_embeddings (torch.Tensor): Tensor of shape (batch_size, num_bits), containing 0s and 1s.

    Returns:
        torch.Tensor: Packed embeddings of shape (batch_size, num_bytes).
    """
    batch_size, num_bits = binary_embeddings.shape
    # Ensure binary_embeddings are uint8
    binary_embeddings = binary_embeddings.round().byte()
    # Calculate the number of bytes needed
    num_bytes = (num_bits + 7) // 8
    # Pad the binary embeddings to make the number of bits a multiple of 8
    padding_bits = num_bytes * 8 - num_bits
    if padding_bits > 0:
        padding = torch.zeros(batch_size, padding_bits, dtype=torch.uint8, device=binary_embeddings.device)
        binary_embeddings = torch.cat((binary_embeddings, padding), dim=1)
    # Reshape to (batch_size, num_bytes, 8)
    binary_embeddings = binary_embeddings.view(batch_size, num_bytes, 8)
    # Pack bits into bytes
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=binary_embeddings.device)
    packed_embeddings = torch.einsum('bni,i->bn', binary_embeddings, weights)
    return packed_embeddings

def hamming_distance(packed_a, packed_b):
    """
    Computes the Hamming distance between two packed embeddings.

    Args:
        packed_a (np.ndarray): Packed embedding, shape (num_bytes,).
        packed_b (np.ndarray): Packed embedding, shape (num_bytes,).

    Returns:
        int: Hamming distance between the two embeddings.
    """
    # Compute XOR
    xor_result = np.bitwise_xor(packed_a, packed_b)
    # Compute Hamming distance using popcount
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    hamming_distance = popcount_table[xor_result].sum()
    return hamming_distance

# Example usage:

# Assuming binary_embeddings is obtained from the QuantizationModule with binary=True
batch_size = 5
num_bits = 23  # Example number of bits
binary_embeddings = torch.randint(0, 2, (batch_size, num_bits), dtype=torch.float32)

# Pack the binary embeddings
packed_embeddings = pack_bits(binary_embeddings)

# Convert to numpy arrays
packed_embeddings_np = packed_embeddings.cpu().numpy()

# Compute Hamming distance between the first two embeddings
distance = hamming_distance(packed_embeddings_np[0], packed_embeddings_np[1])
print(f"Hamming distance between embedding 0 and 1: {distance}")

# Compute Hamming distances between the first embedding and all others
xor_results = np.bitwise_xor(packed_embeddings_np[1:], packed_embeddings_np[0])
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
hamming_distances = popcount_table[xor_results].sum(axis=1)
print(f"Hamming distances between embedding 0 and others: {hamming_distances}")
```

**Explanation of the Code**

- **pack_bits Function**:
  - **Purpose**: Converts a tensor of binary embeddings into a packed uint8 tensor.
  - **Process**:
    - Pads the embeddings to ensure the number of bits is a multiple of 8.
    - Reshapes the embeddings to group bits into sets of 8.
    - Uses `torch.einsum` with weights corresponding to bit positions to pack bits into bytes.

- **hamming_distance Function**:
  - **Purpose**: Computes the Hamming distance between two packed embeddings using bitwise operations.
  - **Process**:
    - Computes the bitwise XOR of the two embeddings to identify differing bits.
    - Uses a precomputed popcount table to count the number of set bits (differences).

**Considerations**

- **Endianness**: Be consistent with the bit ordering when packing and unpacking bits. The weights used in the `pack_bits` function correspond to big-endian bit ordering. Ensure that any unpacking or bit operations are consistent with this ordering.

- **Data Types**: Use appropriate data types (`np.uint8`, `torch.uint8`) to ensure that bit operations work correctly and to save memory.

- **Padding**: The padding bits added during packing should be handled appropriately during similarity computations. If the padding bits are always zero, they don't contribute to the Hamming distance.

- **Batch Processing**: For large-scale retrieval, process embeddings in batches to utilize vectorized operations and reduce computational overhead.

**Advantages of This Approach**

- **Storage Efficiency**: Reduces storage requirements by a factor of 8 compared to storing bits as bytes and even more compared to floating-point representations.

- **Computational Speed**: Bitwise operations and vectorized computations are highly efficient and can be accelerated using optimized libraries and hardware (e.g., SIMD instructions).

- **Scalability**: Suitable for large-scale retrieval tasks where you need to compute similarities between millions of embeddings.

**Conclusion**

By converting your binary embeddings into packed integer representations and utilizing bitwise operations, you can achieve significant storage savings and computational efficiency. This approach is ideal for similarity computations in retrieval and ranking tasks where speed and resource utilization are critical.

If you have any further questions or need assistance with implementing this approach in your specific context, feel free to ask!



**Understanding the Problem:**

You're working with embeddings of dimension 8, where each dimension is considered a high-information dimension. After quantization, each dimension is represented using **3 bits**, resulting in a binary vector of length \( 8 \times 3 = 24 \) bits per embedding.

Your goals are:

1. **Efficient Storage**:
   - **Reduce storage size** from 24 bytes (if storing each bit as a separate uint8) to **3 bytes** per embedding by packing the 24 bits into 3 `uint8` numbers.
   - **Store embeddings compactly** without unnecessary overhead.

2. **Fast Similarity Computation**:
   - **Compute distances** (e.g., Hamming distance) between embeddings using **bitwise operations** like XOR and population count.
   - **Enable efficient retrieval and ranking** in tasks like nearest neighbor search.

**Your Question:**

- *Does the previous method I described achieve this compact storage and efficient computation? If not, how can we modify or improve the method to meet these requirements?*

---

**Answer:**

Yes, it is entirely possible to pack your 24-bit binary embeddings into **3 `uint8` numbers** per embedding, thereby achieving both **efficient storage** and **fast similarity computations** using bitwise operations. Below, I'll explain how to do this step by step.

---

## **Step 1: Quantize Embeddings to Binary Codes**

After the quantization process with `binary=True` in your `QuantizationModule`, you obtain binary embeddings of length 24 bits per embedding.

- **Each embedding dimension (total of 8) is represented by 3 bits**, corresponding to 4 quantization levels (since \( 2^3 = 8 \) but we have levels 0-3 due to 2-bit quantization explained earlier).
- The resulting binary vector has 24 bits.

---

## **Step 2: Pack Bits into Bytes for Efficient Storage**

To store these 24 bits efficiently, we can pack them into 3 bytes (since \( 3 \times 8 = 24 \) bits). Here's how:

### **2.1 Convert Binary Embeddings to Integer Bits**

Ensure that your binary embeddings are in integer form (0 or 1) and of type `numpy.uint8` or `torch.uint8`.

```python
# Convert float tensor to uint8 with values 0 or 1
binary_embeddings = binary_embeddings.round().byte()
```

### **2.2 Pack Bits into Bytes**

**Process:**

- **Group the bits into chunks of 8** (since 1 byte = 8 bits).
- **Pack each group into a single byte** using bit manipulation.

**Implementation:**

```python
import numpy as np

def pack_bits(bits):
    """
    Packs an array of bits into an array of uint8, with 8 bits per uint8 element.

    Args:
        bits (np.ndarray): Array of bits (0 or 1), shape (num_bits,)

    Returns:
        np.ndarray: Packed bits as an array of uint8, shape (num_bytes,)
    """
    bits = np.array(bits, dtype=np.uint8)
    num_bits = bits.size
    num_bytes = (num_bits + 7) // 8  # Ceiling division to get the number of bytes
    padding = num_bytes * 8 - num_bits
    if padding > 0:
        # Pad with zeros to make the number of bits a multiple of 8
        bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
    # Reshape bits into shape (num_bytes, 8)
    bits = bits.reshape(num_bytes, 8)
    # Use np.packbits to pack bits into uint8
    packed = np.packbits(bits, axis=1, bitorder='big')  # 'big' for big-endian
    # Flatten to get a 1D array of bytes
    return packed.flatten()
```

**Example:**

```python
# Example binary embedding of 24 bits
binary_embedding = np.random.randint(0, 2, 24)
packed_embedding = pack_bits(binary_embedding)
print(f"Packed Embedding: {packed_embedding}")  # Should be an array of 3 uint8 elements
```

- **Storage Size:** Each embedding now occupies **3 bytes**, reflecting the actual number of bits used (24 bits).

---

## **Step 3: Store the Packed Embeddings**

Store the packed embeddings in an efficient data structure:

```python
# Suppose we have multiple embeddings
embeddings_list = [np.random.randint(0, 2, 24) for _ in range(num_embeddings)]
packed_embeddings = [pack_bits(emb) for emb in embeddings_list]

# Convert to a numpy array of shape (num_embeddings, 3)
packed_embeddings = np.vstack(packed_embeddings)
```

- **Data Type:** `np.ndarray` with `dtype=np.uint8`
- **Shape:** `(num_embeddings, 3)`

---

## **Step 4: Compute Similarity Using Bitwise Operations**

To compute the Hamming distance between embeddings:

### **4.1 Bitwise XOR**

Compute the bitwise XOR between two packed embeddings:

```python
def compute_hamming_distance(packed_a, packed_b):
    """
    Computes the Hamming distance between two packed bit arrays.

    Args:
        packed_a (np.ndarray): Packed bits of embedding A, shape (num_bytes,)
        packed_b (np.ndarray): Packed bits of embedding B, shape (num_bytes,)

    Returns:
        int: Hamming distance between embedding A and B.
    """
    xor_result = np.bitwise_xor(packed_a, packed_b)
    # Count the number of set bits (1s) in xor_result
    return np.unpackbits(xor_result, bitorder='big').sum()
```

### **4.2 Population Count (Popcount)**

Counting the number of set bits (population count) can be optimized:

- **Use numpy's `np.unpackbits`** function.
- **Utilize a lookup table** for more efficiency if dealing with large datasets.

**Example:**

```python
# Compute Hamming distance between embedding 0 and 1
distance = compute_hamming_distance(packed_embeddings[0], packed_embeddings[1])
print(f"Hamming Distance: {distance}")
```

### **4.3 Vectorized Computation for Multiple Embeddings**

For efficient retrieval:

```python
def compute_hamming_distances(packed_query, packed_dataset):
    """
    Computes Hamming distances between a query embedding and a dataset of embeddings.

    Args:
        packed_query (np.ndarray): Packed bits of the query embedding, shape (num_bytes,)
        packed_dataset (np.ndarray): Packed bits of dataset embeddings, shape (num_embeddings, num_bytes)

    Returns:
        np.ndarray: Hamming distances, shape (num_embeddings,)
    """
    # Perform XOR between the query and all dataset embeddings
    xor_results = np.bitwise_xor(packed_dataset, packed_query)
    # Use np.unpackbits and sum along axis=1 to compute Hamming distances
    hamming_distances = np.unpackbits(xor_results, axis=1, bitorder='big').sum(axis=1)
    return hamming_distances

# Example usage
query_embedding = packed_embeddings[0]
distances = compute_hamming_distances(query_embedding, packed_embeddings[1:])
print(f"Hamming Distances to Query: {distances}")
```

- **Optimized for Large Datasets:** This method is efficient and scales well with the number of embeddings.

---

## **Advantages of This Approach**

- **Storage Efficiency:**
  - Reduces storage from 24 bytes per embedding (if storing each bit separately) to **3 bytes** per embedding.
  - Efficiently utilizes storage by packing bits into bytes.

- **Computation Speed:**
  - Bitwise operations (`XOR`, `AND`, etc.) are highly efficient and can be accelerated using SIMD instructions on modern CPUs.
  - **Hamming distance computations** are fast due to vectorized operations and optimized bit manipulation.

- **Scalability:**
  - Suitable for large-scale retrieval tasks where you need to compute similarities between millions of embeddings.

---

## **Additional Considerations**

### **Endianness and Bit Order**

- **Bit Order Consistency:**
  - Ensure that the bit order is consistent when packing and unpacking bits.
  - Use the `bitorder` parameter (`'big'` or `'little'`) in `np.packbits` and `np.unpackbits`.

### **Padding Bits**

- **Handling Padding:**
  - When the number of bits is not a multiple of 8, padding bits are added.
  - Ensure that these padding bits are handled correctly during similarity computations (typically, they can be ignored or set to zero).

### **Popcount Optimization**

- **Lookup Tables:**
  - For even faster population counts, use a precomputed lookup table for the number of set bits in each possible byte value (0 to 255).

  ```python
  popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
  def compute_hamming_distance_lut(packed_a, packed_b):
      xor_result = np.bitwise_xor(packed_a, packed_b)
      return popcount_table[xor_result].sum()
  ```

### **Libraries and Tools**

- **Use Specialized Libraries:**
  - Consider using libraries like **FAISS** (Facebook AI Similarity Search) for efficient similarity search on large datasets. FAISS supports binary vectors and provides optimized functions for similarity computation.

  - **Example:** FAISS has a `IndexBinaryFlat` that can index and search over binary vectors using Hamming distance.

---

## **Implementation Example**

Here's a complete example incorporating all the steps:

```python
import numpy as np

def pack_bits(bits):
    bits = np.array(bits, dtype=np.uint8)
    num_bits = bits.size
    num_bytes = (num_bits + 7) // 8
    padding = num_bytes * 8 - num_bits
    if padding > 0:
        bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
    bits = bits.reshape(num_bytes, 8)
    packed = np.packbits(bits, axis=1, bitorder='big')
    return packed.flatten()

def compute_hamming_distance(packed_a, packed_b, popcount_table):
    xor_result = np.bitwise_xor(packed_a, packed_b)
    return popcount_table[xor_result].sum()

def compute_hamming_distances(packed_query, packed_dataset, popcount_table):
    xor_results = np.bitwise_xor(packed_dataset, packed_query)
    distances = popcount_table[xor_results].sum(axis=1)
    return distances

# Prepare popcount lookup table
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# Example embeddings
embedding_dim = 8
bits_per_dim = 3
total_bits = embedding_dim * bits_per_dim  # 24 bits

# Generate random embeddings
num_embeddings = 1000  # Example dataset size
np.random.seed(42)
embeddings_bits = np.random.randint(0, 2, (num_embeddings, total_bits), dtype=np.uint8)

# Pack embeddings
packed_embeddings = np.array([pack_bits(bits) for bits in embeddings_bits])

# Select a query embedding
query_embedding = packed_embeddings[0]

# Compute Hamming distances to all embeddings in the dataset
distances = compute_hamming_distances(query_embedding, packed_embeddings, popcount_table)

# Retrieve top-k similar embeddings (smallest Hamming distances)
k = 5
top_k_indices = np.argsort(distances)[:k]
print(f"Top-{k} similar embeddings indices: {top_k_indices}")
print(f"Corresponding Hamming distances: {distances[top_k_indices]}")
```

**Explanation:**

- **Packing Bits:**
  - Each embedding is represented by 24 bits and packed into 3 bytes.
  - The dataset is a NumPy array of shape `(num_embeddings, 3)`.

- **Computing Distances:**
  - **compute_hamming_distance:** Computes the Hamming distance between two embeddings using the popcount lookup table.
  - **compute_hamming_distances:** Computes Hamming distances between the query and all dataset embeddings efficiently.

- **Retrieval:**
  - Retrieves the indices of the top-k embeddings with the smallest Hamming distances to the query.

---

## **Summary**

- **Your methods can achieve the desired storage and computational efficiency** by ensuring that:

  - **Bits are properly packed** into the minimum number of bytes required.
  - **Bitwise operations are used** for similarity computations.
  - **Optimizations** like lookup tables are implemented for faster computations.

- **Implementing these strategies will allow you to:**

  - **Store each embedding in just 3 bytes** instead of 24 bytes.
  - **Perform fast similarity computations** for retrieval and ranking.

---

# Training for Quants
we want to store our embeddings as 1 bit per dimension (just 0 or 1) for each dimension. How can we train a layer on top of existing embeddings (or add this to the already done MRL training) to ensure that our embeddings do well when quantised to binary per dimension. What are the considerations and what training methods do we need to follow.

In the QuantizationLayer, thresholds for multi-bit quantization should be properly initialized, especially since 2-bit quantization requires 3 thresholds.

Implement an initialization method that calculates initial thresholds based on sample embeddings. Implement proper initialization of thresholds for 2-bit quantization based on sample embeddings. To utilize the new initialization method, you need to collect sample embeddings before initializing the MatryoshkaEmbeddingModel.

1. **Implement Threshold Initialization**:
   - Create an `initialize_thresholds` method in the `QuantizationLayer`.
   - This method computes initial thresholds based on provided sample embeddings.
   - Update the training code to collect sample embeddings and call this method after the model is instantiated.



<obsolete_sigmoid_temperature_annealing>


**Summary of Our Work on the Matryoshka Embedding Model and Quantization Techniques**

---

Throughout our discussions, we've worked extensively on enhancing the **Matryoshka Embedding Model** by integrating advanced quantization techniques, specifically **1-bit (binary)** and **2-bit quantization**. Our goal was to develop efficient embedding representations that reduce storage requirements and computational complexity while maintaining high similarity search performance.

Below, I'll provide a detailed explanation of the work we've done, including the mathematical formulations, the structure of our code, and how these quantization methods enable effective similarity computations and nearest neighbor searches.

---

### **1. Matryoshka Embedding Model**

The **Matryoshka Embedding Model** is inspired by the concept of Russian nesting dolls, where embeddings are nested within each other at varying dimensions. This model allows us to generate embeddings at multiple scales, providing flexibility in terms of storage and computational requirements.

**Key Features:**

- **Hierarchical Embeddings:** Produces embeddings of different dimensions (e.g., 256, 512, 1024) that are subsets of each other.
- **Efficiency:** Smaller embeddings capture essential information, while larger embeddings include finer details.
- **Flexibility:** Allows for deployment across devices with varying resource constraints.

---

### **2. Quantization Layer**

The **QuantizationLayer** is a critical component that quantizes continuous embeddings into discrete representations. We've incorporated two types of quantization:

- **1-Bit Quantization (Binary Quantization):** Each embedding dimension is represented by a single bit.
- **2-Bit Quantization:** Each embedding dimension uses two bits, allowing for four discrete levels.

**Role of the QuantizationLayer:**

- Reduces the size of embeddings significantly.
- Enables the use of efficient similarity computation methods suitable for quantized data.
- Integrates seamlessly with the Matryoshka Embedding Model for multi-scale quantization.

---

### **3. 1-Bit Quantization**

**Implementation Details:**

- **Thresholding:** We apply a learned threshold \( t \) to each embedding dimension to determine the binary value.
- **Sigmoid Function:** During training, we use a sigmoid activation to allow gradients to flow.
- **Straight-Through Estimator (STE):** Enables gradient computation through discrete operations.

**Mathematical Formulation:**

Let \( x \in \mathbb{R}^d \) be the input embeddings, and \( t \in \mathbb{R}^d \) be the learnable thresholds.

1. **Compute Pre-Activation Values:**

   \[
   z = x - t
   \]

2. **Apply Sigmoid Function with Temperature Scaling:**

   \[
   \sigma(z) = \frac{1}{1 + e^{-z / T}}
   \]

   - \( T \) is the temperature parameter controlling the sigmoid's steepness.

3. **Quantized Output During Training:**

   \[
   q = \sigma(z)
   \]

4. **Binary Output During Inference:**

   \[
   q_{\text{binary}} = \begin{cases}
   1, & \text{if } z > 0 \\
   0, & \text{otherwise}
   \end{cases}
   \]

**Temperature Annealing:**

- The temperature \( T \) is **annealed** over epochs to transition from smooth to sharp sigmoid functions.
- Annealing schedule ensures \( T \) decreases from \( T_0 \) to \( T_{\text{min}} \) over the total number of epochs.

---

### **4. 2-Bit Quantization**

**Implementation Details:**

- **Multi-Threshold Quantization:** Uses multiple thresholds to divide the embedding space into discrete levels.
- **Sigmoid Functions:** Softly assign probabilities to each quantization level during training.
- **Codebook Mapping:** Optionally expands 2-bit codes to 3-bit representations for efficient similarity computation.

**Mathematical Formulation:**

1. **Thresholds:**

   - Learnable thresholds \( \{ t_1, t_2, t_3 \} \) for each dimension.

2. **Compute Logits for Sigmoid Functions:**

   \[
   \text{logits}_i = \frac{x - t_i}{T}, \quad i = 1, 2, 3
   \]

3. **Compute Sigmoid Activations:**

   \[
   \sigma_i = \frac{1}{1 + e^{-\text{logits}_i}}, \quad i = 1, 2, 3
   \]

4. **Calculate Probabilities for Each Quantization Level:**

   \[
   \begin{align*}
   p_0 & = 1 - \sigma_1 \\
   p_1 & = \sigma_1 - \sigma_2 \\
   p_2 & = \sigma_2 - \sigma_3 \\
   p_3 & = \sigma_3
   \end{align*}
   \]

5. **Expected Quantized Output:**

   \[
   q = \sum_{i=0}^{3} i \cdot p_i
   \]

6. **Discrete Quantization Using STE During Training:**

   \[
   q_{\text{quantized}} = q + ( \text{round}(q) - q ).\text{detach}()
   \]

**Threshold Initialization:**

- Thresholds are initialized based on sample embeddings using statistical measures (e.g., quantiles).
- Proper initialization improves convergence and quantization performance.

---

### **5. Temperature Parameter and Annealing Schedule**

**Purpose of Temperature \( T \):**

- Controls the **steepness** of the sigmoid functions in quantization.
- Higher \( T \): Flatter sigmoid, smoother quantization assignments.
- Lower \( T \): Steeper sigmoid, sharper transitions, more discrete outputs.

**Annealing Schedule:**

To ensure the temperature decreases from \( T_0 \) to \( T_{\text{min}} \) over the total number of epochs \( N \):

1. **Calculate Annealing Rate \( r \):**

   \[
   r = \left( \frac{T_{\text{min}}}{T_0} \right)^{1/N}
   \]

2. **Update Temperature at Epoch \( n \):**

   \[
   T_n = T_0 \times r^n
   \]

This schedule ensures that \( T_N = T_{\text{min}} \) at the final epoch.

---

### **6. Similarity Computation and Nearest Neighbor Search**

**Challenges with Quantized Embeddings:**

- Traditional similarity measures (e.g., cosine similarity) operate on continuous vectors.
- Quantized embeddings require adapted methods for similarity computation.

**Solutions:**

- **Binary Embeddings:**
  - Use **Hamming Distance** or **Bitwise Operations** for efficient similarity computation.
  - Hamming Distance \( d_H(a, b) \): Counts differing bits between binary vectors \( a \) and \( b \).

- **2-Bit Embeddings:**
  - Expand 2-bit codes to higher-bit representations (e.g., 3-bit codes) to facilitate similarity measures.
  - Precompute **Lookup Tables** for efficient computation.

**Approximate Nearest Neighbor (ANN) Search:**

- Utilize data structures and algorithms optimized for quantized data (e.g., **Product Quantization**, **LSH**).
- ANN methods provide significant speed-ups in large-scale similarity search tasks.

---

### **7. Code Structure and Implementation Details**

Our code is structured to be modular, maintainable, and extensible. Below is an overview of the main components and how they interact.

#### **Classes and Their Roles:**

- **MatryoshkaEmbeddingModel:**

  - Inherits from a base embedding model.
  - Integrates the **QuantizationLayer** for quantization.
  - Handles encoding of sentences into embeddings at various dimensions.
  - Methods:
    - `__init__`: Initializes the model with specified configurations.
    - `encode`: Encodes input text and applies quantization if enabled.
    - `init_thresholds`: Initializes thresholds using sample embeddings.
    - `save` / `load`: Handles model persistence.

- **MatryoshkaTransformer:**

  - A neural network that transforms base embeddings into nested embeddings.
  - Contains multiple blocks corresponding to the different dimension levels.

- **QuantizationLayer:**

  - Performs quantization of embeddings.
  - Supports both 1-bit and 2-bit quantization.
  - Contains methods for forward computation, temperature annealing, and threshold initialization.

#### **Key Implementation Details:**

- **Modularity:**

  - Quantization features are controlled via flags (`train_binary`, `train_two_bit`).
  - Ensures that adding new quantization methods doesn't interfere with existing functionality.

- **Temperature and Threshold Management:**

  - **Temperature** is registered as a buffer to track its state across devices and save/load operations.
  - **Thresholds** are learnable parameters initialized based on sample data.

- **Training Loop Adjustments:**

  - Collects sample embeddings to initialize thresholds before training starts.
  - Anneals temperature at each epoch to guide the model from smooth to discrete outputs.
  - Implements regularization losses for quantization and embedding quality.

- **Loss Functions:**

  - **Multi-Scale Contrastive Loss:**
    - Encourages similarity between transformed embeddings and their originals across dimensions.
  - **Quantization Regularization Loss:**
    - Encourages embeddings to be close to discrete quantization levels.
  - **Orthogonality Regularization:**
    - Ensures new dimensions capture unique information.
  - **Information Bottleneck Regularization:**
    - Encourages lower dimensions to retain critical information.

---

### **8. Additional Contributions and Enhancements**

**Orthogonality Regularization:**

- **Purpose:** Prevents redundancy by ensuring newly added dimensions are orthogonal to previous ones.
- **Implementation:**
  - Minimizes the dot product between the new dimensions and existing ones.

**Information Bottleneck Regularization:**

- **Purpose:** Encourages the model to capture essential information in lower dimensions.
- **Implementation:**
  - Applies progressively stronger regularization to higher dimensions.

**Integration of Multi-Bit Quantization:**

- **Expand to 3-Bit Codes:**

  - Implements a codebook to map 2-bit quantized values to 3-bit representations.
  - Facilitates efficient similarity computations using bitwise operations.

**Hyperparameter Configurations:**

- **Adaptive Annealing Rate:**

  - Calculated based on total epochs to ensure the temperature reaches \( T_{\text{min}} \) precisely at the end of training.

- **Proper Threshold Initialization:**

  - Uses statistical methods (e.g., percentiles) to set initial thresholds, enhancing convergence.

---

### **9. Overall Workflow and Training Procedure**

1. **Initialization:**

   - Instantiate the `MatryoshkaEmbeddingModel` with desired configurations.
   - Collect sample embeddings to initialize thresholds in the `QuantizationLayer`.

2. **Training Loop:**

   - For each epoch:
     - Forward pass through the model to obtain embeddings at various dimensions.
     - Compute loss functions (contrastive loss, regularization losses).
     - Backpropagate and update model parameters.
     - Anneal the temperature parameter.

3. **Quantization Regularization:**

   - Applied during training to ensure embeddings conform to quantized levels.

4. **Evaluation:**

   - Quantized embeddings are tested for similarity search tasks.
   - Performance compared against baseline models.

---

### **10. Benefits and Applications**

- **Storage Efficiency:**

  - Significant reduction in embedding size (e.g., from 32 bits per dimension to 1 or 2 bits).

- **Computational Speed:**

  - Faster similarity computations using bitwise operations and specialized data structures.

- **Scalability:**

  - Enables handling large-scale datasets and embedding collections efficiently.

- **Flexibility in Deployment:**

  - Different embedding dimensions can be utilized based on resource availability and performance requirements.

---

### **Conclusion**

Our work has focused on enhancing the Matryoshka Embedding Model by integrating sophisticated quantization techniques, specifically 1-bit and 2-bit quantization. By carefully designing the **QuantizationLayer** and implementing strategies like proper threshold initialization and temperature annealing, we've developed a model that efficiently produces quantized embeddings suitable for similarity search tasks.

The modular and configurable nature of our code allows for easy adaptation and extension. Through rigorous mathematical formulations and diligent implementation, we've addressed key challenges in quantization, paving the way for practical applications that demand efficient and scalable embedding representations.

---


We have also discussed how we did 3-bit quantization wherein we make quantization to 2 bits and then expand the 2 bits to 3 bits to ensure that we can use bitwise operations for computing similarity. 

```
codebook = {  
            0: torch.tensor([0, 0, 0], dtype=torch.float32),  
            1: torch.tensor([0, 0, 1], dtype=torch.float32),  
            2: torch.tensor([0, 1, 1], dtype=torch.float32),  
            3: torch.tensor([1, 1, 1], dtype=torch.float32)  
        }  
        return codebook  
```

The codebook described above converts a 2-bit quantized value to a 3-bit quantized value and then on this 3-bit quantized value direct bitwise operations can give the distance.


### Comprehensive Summary of Our Work

**Objective:**
Our primary objective was to enhance the efficiency and performance of the **Matryoshka Embedding Model** by incorporating quantization strategies. Quantization allows for reducing the memory footprint and computation time by compressing continuous-valued embeddings into discrete representations.

**Key Techniques Implemented:**

1. **1-Bit Quantization:**
    - **Purpose:** Reduce each embedding dimension to a single bit.
    - **Method:** 
        - Learn thresholds to classify each dimension into binary outputs (0 or 1).
        - Use a sigmoid activation during training for smooth gradient flow.
        - Apply a Straight-Through Estimator (STE) to handle non-differentiability.

2. **2-Bit Quantization:**
    - **Purpose:** Represent each dimension with two bits, allowing four discrete levels.
    - **Method:**
        - Set multiple thresholds to achieve a finer quantization scale.
        - Compute probabilities for each level using the sigmoid function modified by a temperature parameter.
        - Use codebook mapping to expand 2-bit codes to 3-bit for efficient similarity computation.

3. **3-Bit Expansion via Codebook:**
    - **Purpose:** Enable direct bitwise operations for similarity measures.
    - **Codebook Implementation:** 
        - Map each 2-bit code (0, 1, 2, 3) to a corresponding 3-bit binary vector.
        - Facilitates operations like Hamming distance computation directly on quantized embeddings.

**Annealing Temperature:**
- **Reason:** To control the steepness of the sigmoid activation.
- **Implementation:** Start with a high temperature for smooth gradients, decreasing it over epochs to enforce discrete outputs.

**Benefits:**
- **Memory Efficiency:** Reduction in storage requirements for embeddings, crucial for large-scale applications.
- **Computation Speed:** Faster operations due to simplified data representation.
- **Simplified Similarity Computation:** Using bitwise operations to measure similarity accelerates processes like nearest neighbor search.

---

### Detailed Report from a PyTorch Developer's Perspective

As a PyTorch developer, our journey to integrate robust quantization techniques into the Matryoshka Embedding Model reflects both a challenge and an opportunity to push the boundaries of efficient machine learning deployment. This report outlines our methodology and the mechanics of our approach.

#### **Methodology and Technical Implementation**

**Matryoshka Embedding Model Overview:**
- **Architecture:** Utilizes a nested approach where embeddings are scaled across different dimensional hierarchies, allowing flexible deployment depending on resource constraints. This hierarchical structure is particularly adept at balancing performance and efficiency.

**Quantization Techniques:**

1. **1-Bit Quantization:**
    - **Activation and Thresholding:** The PyTorch implementation leverages tensor operations to efficiently broadcast thresholds across input embeddings, applying sigmoid activation functions during training:
      ```python
      x = x - self.thresholds
      x = torch.sigmoid(x / self.temperature)
      ```
    - **Binary Conversion:** During evaluation, embeddings are converted to a strict binary form using thresholding:
      ```python
      x = (x > 0).float()
      ```
    - **Maintained Gradients:** By employing a Straight-Through Estimator (STE), we seamlessly integrate discrete operations within the computational graph, facilitating gradient-based optimization.

2. **2-Bit Quantization:**
    - **Multi-Threshold Setup:** By configuring and sorting thresholds as learnable parameters, we ensure dynamic adaptation during training.
      ```python
      logits = (embeddings - thresholds) / self.temperature
      sigma = torch.sigmoid(logits)
      ```
    - **Computing Quantization Levels:** The code effectively computes probabilities for each discrete level using tensor operations, making the PyTorch backend ideal for these vectorized calculations.

3. **3-Bit Expansion:**
    - **Codebook Mapping:** Implemented via a simple dictionary structure; this Pythonic approach ensures clarity and efficiency.
      ```python
      codebook = {
          0: torch.tensor([0, 0, 0]),
          1: torch.tensor([0, 0, 1]),
          2: torch.tensor([0, 1, 1]),
          3: torch.tensor([1, 1, 1])
      }
      ```
    - **Exact Mapping for Simplicity:** By expanding 2-bit values to 3-bit values, we align the embedding format with bitwise operation capabilities, leveraging intrinsic PyTorch methods.

**Temperature Annealing:**
- **Controlled Decay:** Using PyTorch's tensor operations, we methodically decrease `self.temperature`:
  ```python
  r = (self.min_temperature / self.initial_temperature) ** (1 / total_epochs)
  new_temperature = self.initial_temperature * (r ** current_epoch)
  ```

**Model Training and Optimization:**
- **Loss Computation:** Integrates contrastive loss for multi-scale embeddings with custom regularization terms, ensuring the embeddings preserve their hierarchical structure while conforming to quantized levels.
- **Gradient Clipping and Optimization:** Employed to stabilize training dynamics, supported by PyTorch's inbuilt utilities, `torch.optim` for adaptive learning.

#### **Code Structure and Flow:**

1. **Class `MatryoshkaEmbeddingModel`:**
   - Manages the overall workflow, handling initialization, encoding, and threshold setup.
   - Ensures modularity by allowing toggling between quantization modes.

2. **Class `QuantizationLayer`:**
   - Encapsulates quantization logic, maintaining a focus on learnability and adaptability.
   - Houses the threshold learning mechanism and the codebook expansion logic.

3. **Training Procedure:**
   - **Initialization and Threshold Collection:** Efficiently gathers initial data samples to set robust thresholds.
   - **Epoch-Based Training Loop:** Iteratively refines embeddings, anneals temperature, and updates thresholds.
---



## Summary of Our Work on Matryoshka Embedding Model with Quantization

### Main Enhancements and Objectives

Our work aims to enhance the **Matryoshka Embedding Model** through advanced quantization techniques, specifically leveraging 1-bit, 2-bit, and 3-bit expansions. The primary goals are to reduce memory and computational costs while maintaining high performance in similarity computation tasks.

### Quantization Techniques

#### 1-Bit Quantization

- **Objective:** Compress each embedding dimension to a single bit.
- **Methodology:**
  - **Threshold Learning:** A threshold \\( t \\) per dimension learns to convert continuous values to binary outputs.
  - **Sigmoid Activation:** Applied during training to allow gradients to propagate.
  - **Straight-Through Estimator (STE):** Facilitates gradient flow through the non-differentiable step of binary decisions.
  
  **Mathematical Formulation:**
  \[
  \sigma(z) = \frac{1}{1 + e^{-\frac{z}{T}}}
  \]
  - **Binary Conversion:** During evaluation:
  \[
  q_{\text{binary}} = \begin{cases} 
  1, & \text{if } z > 0 \\
  0, & \text{otherwise} 
  \end{cases}
  \]

#### 2-Bit Quantization

- **Objective:** Allow more detailed quantization with two bits per dimension (four levels).
- **Methodology:**
  - **Multi-Threshold Quantization:** Uses multiple thresholds \\( \{ t_1, t_2, t_3 \} \\) to segment the input space.
  - **Probabilities Computation:** Using sigmoid functions to softly assign probabilities to each quantization level.
  
  **Mathematical Formulation:**
  \[
  p_i = \sigma_i - \sigma_{i+1} 
  \]
  - **Quantized Output:** 
  \[
  q = \sum_{i=0}^{3} i \cdot p_i
  \]

#### 3-Bit Quantization via Expansion

- **Objective:** Facilitate efficient similarity computation using bitwise operations.
- **Implementation:**
  - **Codebook Mapping:** Maps each 2-bit code to a 3-bit representation:
  ```python
  codebook = {
      0: torch.tensor([0, 0, 0], dtype=torch.float32),
      1: torch.tensor([0, 0, 1], dtype=torch.float32),
      2: torch.tensor([0, 1, 1], dtype=torch.float32),
      3: torch.tensor([1, 1, 1], dtype=torch.float32)
  }
  ```

### Temperature Annealing

- **Purpose:** Gradually reduce the temperature to ensure smoother training transitioning to sharp quantizations.
- **Formula for Annealing Rate \\( r \\):** 
  \[
  r = \left( \frac{T_{\text{min}}}{T_0} \right)^{\frac{1}{N}}
  \]

### Code Structure and Flow

- **`MatryoshkaEmbeddingModel`:** Centralizes the encoding and quantization processes, managing model configuration and embeddings.
- **`QuantizationLayer`:** Handles the quantization logic, including forward calculation and temperature handling.
- **`MatryoshkaTransformer`:** Applies transformation rules to generate nested embeddings.
  
### Benefits and Applications

- **Efficiency:** The quantization methods significantly reduce the memory footprint of embeddings.
- **Speed:** Facilitates fast similarity computations with bitwise operations.
- **Flexibility:** Supports applications across various resource limitations.

---

## Technical Report: Perspective of a PyTorch Developer

### Introduction

The integration of quantization techniques with the Matryoshka Embedding Model represents a substantial advancement in neural network efficiency, balancing memory savings with high performance. This report captures the implementation details and benefits from a PyTorch developer's perspective.

### Technical Implementation

#### Quantization Techniques

1. **1-Bit Quantization:**
   - Utilizes tensor operations to efficiently apply learned thresholds to each dimension:
   ```python
   x = x - self.thresholds
   x = torch.sigmoid(x / self.temperature)
   ```

2. **2-Bit Quantization:**
   - Handles multi-threshold logic dynamically adjusting thresholds as learnable parameters during training:
   ```python
   logits = (embeddings - thresholds) / self.temperature
   sigma = torch.sigmoid(logits)
   ```
   - Computes and maps quantized levels effectively using PyTorch's vectorized computation capabilities.

3. **3-Bit Expansion:**
   - Simplifies bitwise operation handling with a clear codebook translation:
   ```python
   def expand_to_three_bits(self, quantized_embeddings: torch.Tensor) -> torch.Tensor:
       ...
   ```

#### Code Structure

1. **`MatryoshkaEmbeddingModel`:**
   - Manages initialization and threshold configuration, supporting various quantization types.

2. **`QuantizationLayer`:**
   - Encapsulates quantization methodology, supporting flexibility in training and inference scenarios.

3. **Training Dynamics:**
   - Processes sample embeddings to initialize thresholds, applying annealing, and executing optimization routines.

## Sigmoid Temperature Annealing code summary


Certainly! I'd be happy to provide a detailed summary of your code, explaining how it works, the flow of execution, and how each part contributes to achieving your goals of efficient embedding quantization in the Matryoshka Embedding Model.

---

## **Overview**

The provided code implements the **Matryoshka Embedding Model** with advanced quantization techniques, specifically focusing on 1-bit and 2-bit quantization. The main objectives are:

- **Memory Efficiency**: Reduce the size of embeddings by quantizing them into lower bits.
- **Computational Efficiency**: Enable faster similarity computations using quantized embeddings.
- **Scalability**: Produce nested embeddings that can be adjusted based on resource constraints.

The code is organized into several key classes:

1. **`MatryoshkaEmbeddingModel`**: The main model class that integrates the embedding model with the transformer and handles encoding.
2. **`MatryoshkaTransformer`**: Applies transformations to produce nested embeddings and applies quantization layers.
3. **`QuantizationLayer`**: Performs quantization of embeddings (1-bit or 2-bit) with temperature scaling and thresholding.
4. **`SliceLayer`**: A helper layer that slices the tensor along the dimension to obtain embeddings of specified sizes.

---

## **Detailed Breakdown**

### **1. MatryoshkaEmbeddingModel Class**

**Purpose**: Integrates the base embedding model with the transformer and handles the encoding process, including quantization.

**Key Attributes**:

- `embedding_model`: The pretrained embedding model (e.g., SentenceTransformer).
- `dimension_levels`: List of embedding dimensions in increasing order (e.g., `[256, 512, 1024]`).
- `train_binary`, `train_two_bit`, `expand_two_bit_to_three_bits`: Flags to control the quantization training modes.
- `transformer`: An instance of `MatryoshkaTransformer` that generates nested embeddings.
- `max_dim`: The maximum embedding dimension.
- `baseline`: A flag to use baseline embeddings without transformation.

**Methods**:

- `__init__(...)`: Initializes the model, sets up the transformer with quantization layers based on the provided flags.
- `save(path)`: Saves the state of the transformer to a file.
- `load(path)`: Loads the transformer state from a file.
- `calculate_thresholds(sample_embeddings)`: Calculates thresholds for quantization based on sample embeddings.
- `init_thresholds(sample_embeddings)`: Initializes thresholds in the quantization layers.
- `encode(sentences, output_dim, **kwargs)`: Encodes sentences into embeddings of the specified dimension, applying quantization if needed.

**Encoding Flow**:

1. **Get Base Embeddings**:

   - Uses the `embedding_model` to encode sentences into base embeddings.
   - The embeddings are not normalized at this stage.

2. **Transformation**:

   - Passes the base embeddings through the `transformer` to get nested embeddings at various dimensions.
   - If `baseline` is `True`, uses the base embeddings directly.

3. **Quantization**:

   - Depending on the flags `do_binary` and `do_two_bits`, applies quantization to obtain binary or 2-bit embeddings.
   - If quantization is applied, embeddings are further processed to match the expected output dimension, especially if expanding 2-bit codes to 3-bit representations.

4. **Normalization**:

   - If embeddings are not quantized, they are normalized (L2 norm) to ensure they have unit length, which is important for similarity computations.

**Key Points in Achieving Goals**:

- **Flexible Embedding Sizes**: By supporting multiple dimensions, the model can generate embeddings suitable for different resource constraints.
- **Quantization Integration**: The `encode` method seamlessly incorporates quantization, allowing for memory-efficient embeddings.

---

### **2. MatryoshkaTransformer Class**

**Purpose**: Transforms base embeddings to produce nested embeddings at different dimensions. It also integrates quantization layers.

**Key Attributes**:

- `dimension_levels`: List of embedding dimensions.
- `train_binary`, `train_two_bit`, `expand_two_bit_to_three_bits`: Flags controlling quantization modes.
- `blocks`: A `ModuleList` containing the transformation blocks for each dimension level.
- `quantization_layers`: A `ModuleDict` containing the quantization layers for each dimension.
- `base_transform`: A base transformation applied to the input embeddings before dimension-specific blocks.

**Initialization (`__init__`)**:

- **Base Transformation**:

  - Constructs a `nn.Sequential` model that transforms the input embeddings.
  - Consists of `Linear` layers, an activation function (`GELU`), and normalization (`RMSNorm` if `use_rms_norm` is `True`).

  ```python
  self.base_transform = nn.Sequential(
      nn.Linear(input_dim, input_dim * 32),
      nn.GELU(),
      nn.RMSNorm(input_dim * 32, eps=1e-6),  # Optional
      nn.Linear(input_dim * 32, input_dim)
  )
  ```

- **Blocks and Quantization Layers**:

  - For each dimension level, a block is created using `SliceLayer` to select the relevant dimensions.
  - If quantization is enabled, a `QuantizationLayer` is created for that dimension.

**Forward Method (`forward`)**:

1. **Base Transformation**:

   - Applies `self.base_transform` to the input embeddings `x`.

2. **Nested Embeddings Generation**:

   - Iteratively constructs embeddings for each dimension level.
   - For each block:

     - Applies the block to the transformed embeddings to get a delta.
     - Concatenates the delta to the previous embeddings to build up the dimensions incrementally.
     - Normalizes the embeddings using L2 normalization.

3. **Quantization**:

   - If quantization is enabled, passes the embeddings through the corresponding `QuantizationLayer`.
   - Stores both quantized and non-quantized embeddings in separate dictionaries (`embeddings` and `non_quant_embeddings`).

**Key Points in Achieving Goals**:

- **Nested Embeddings**: The hierarchical structure allows the model to output embeddings at multiple scales, facilitating resource-adaptive deployment.
- **Integration with Quantization**: By incorporating `QuantizationLayer` instances, the transformer directly supports quantized representations.

---

### **3. QuantizationLayer Class**

**Purpose**: Performs quantization of embeddings into lower-bit representations (1-bit or 2-bit). It handles temperature scaling, thresholding, and optional expansion of 2-bit codes to 3-bit codes.

**Key Attributes**:

- `dim`: Dimension of the input embeddings.
- `quantization_bits`: Number of bits to quantize (1 or 2).
- `temperature`: Temperature parameter controlling the smoothness of the sigmoid function during training.
- `min_temperature`: The minimum temperature value for annealing.
- `initial_temperature`: Starting temperature value.
- `thresholds`: Learnable thresholds used for quantization.
- `thresholds_0_100`: Stores the 0th and 100th percentiles of the embeddings for reference.
- `codebook`: A mapping from 2-bit codes to 3-bit representations, used if expanding 2-bit codes to 3 bits.
- `annealing_rate`: Rate at which the temperature is annealed during training.

**Methods**:

- `num_thresholds()`: Calculates the number of thresholds based on the number of quantization bits.
- `calculate_thresholds(sample_embeddings)`: Computes thresholds based on the quantiles of the sample embeddings.
- `initialize_thresholds(sample_embeddings)`: Initializes the thresholds using sample embeddings for better convergence.
- `update_thresholds(batch_embeddings, momentum)`: Updates thresholds during training to adapt to new data.
- `create_codebook()`: Creates the codebook for expanding 2-bit codes to 3-bit representations.
- `forward(x, training, quantize_binary, quantize_two_bit)`: Applies quantization to the embeddings during forward pass.
- `multi_bit_quantization(embeddings, training, quantize_two_bit)`: Performs multi-bit quantization with hard thresholding and STE.
- `expand_to_three_bits(quantized_embeddings)`: Expands 2-bit quantized embeddings to 3-bit codes using the codebook.
- `anneal_temperature(current_epoch, total_epochs, num_current_step, num_total_steps, steps_per_epoch)`: Anneals the temperature parameter over training steps.

**Quantization Logic**:

- **1-Bit Quantization**:

  - **Thresholding**: Uses a single threshold to determine if each embedding dimension should be quantized to 0 or 1.
  - **Training Mode**:

    - Uses hard assignments for forward pass.
    - Applies STE to allow gradients to pass through the thresholding operation.

  - **Inference Mode**:

    - Directly applies the thresholding to produce binary embeddings.

- **2-Bit Quantization**:

  - **Thresholding**: Uses three thresholds to divide the embedding values into four levels (0 to 3).
  - **Training Mode**:

    - Similar to 1-bit quantization, but with multiple levels.
    - Applies hard assignments with STE for gradient flow.

  - **Inference Mode**:

    - Uses hard thresholding to quantize embeddings into discrete levels.

- **Expansion to 3 Bits**:

  - If `expand_two_bit_to_three_bits` is `True`, the 2-bit codes are mapped to 3-bit representations using the `codebook`.
  - This expansion allows efficient similarity computations using bitwise operations.

**Temperature Annealing**:

- The temperature parameter controls the smoothness of the sigmoid activation used in quantization.
- Annealing helps transition from a softer quantization (useful for gradient flow during training) to a harder, more discrete quantization during inference.
- The temperature is gradually reduced following an exponential decay schedule.

  $$ T_n = T_0 \times r^{n} $$

  Where:

  - \( T_0 \) is the initial temperature,
  - \( T_n \) is the temperature at epoch \( n \),
  - \( r \) is the annealing rate,
  - \( n \) is the current epoch.

**Key Points in Achieving Goals**:

- **Memory Efficiency**: Quantization reduces the number of bits needed to represent embeddings.
- **Computational Efficiency**: Lower-bit representations and bitwise operations speed up similarity computations.
- **Gradient Flow**: Use of STE and temperature scaling ensures that the model can be trained effectively despite the non-differentiable quantization operations.

---

### **4. SliceLayer Class**

**Purpose**: A simple layer that slices the embeddings along the dimension to obtain a subset embedding of the specified size. This is used in the `MatryoshkaTransformer` to incrementally build up the embeddings for each dimension level.

**Methods**:

- `__init__(start_dim, end_dim)`: Initializes the slice layer with the starting and ending dimensions.
- `forward(x)`: Returns the slice of the input tensor from `start_dim` to `end_dim`.

**Key Points**:

- Helps in constructing embeddings of varying dimensions required for the nested structure of the Matryoshka model.
- Ensures that each block in the transformer works with the appropriate segment of the embedding vector.

---

## **Flow of Execution**

1. **Initialization**:

   - The `MatryoshkaEmbeddingModel` is initialized with a base embedding model and desired dimension levels.
   - Depending on the flags, quantization layers are set up in the transformer.

2. **Encoding Sentences**:

   - The `encode` method is called with sentences and desired output dimension.
   - Base embeddings are obtained from the `embedding_model`.

3. **Transformation and Nested Embeddings**:

   - Base embeddings pass through the `MatryoshkaTransformer`, undergoing the base transformation and being split into nested embeddings of increasing dimensions.

4. **Quantization**:

   - If quantization is enabled, embeddings at each dimension are passed through the corresponding `QuantizationLayer`.
   - Quantization layers apply either 1-bit or 2-bit quantization, possibly expanding 2-bit codes to 3-bit codes.

5. **Post-processing**:

   - Embeddings are normalized if not quantized.
   - The final embeddings are returned as NumPy arrays.

---

## **Achieving Goals with Code Components**

- **Memory Efficiency**:

  - **QuantizationLayer** reduces the number of bits per embedding dimension.
  - **Expansion to 3 Bits** allows using compact representations while facilitating efficient computations.

- **Computational Efficiency**:

  - Quantized embeddings enable faster similarity computations using bitwise operations.
  - Temperature annealing ensures that the model learns to produce embeddings that are conducive to quantization.

- **Scalability and Flexibility**:

  - **MatryoshkaTransformer** produces embeddings at multiple scales, allowing for dynamic adjustment based on resource availability.
  - **SliceLayer** aids in managing embeddings of different dimensions seamlessly.

- **Training Effectiveness**:

  - Use of **Straight-Through Estimator** in `QuantizationLayer` allows backpropagation through the quantization process.
  - **Threshold Initialization** based on sample embeddings ensures that quantization thresholds are appropriately set, facilitating better convergence.

---

## **Mathematical Concepts and Formulations**

### **1. Quantization**

- **1-Bit Quantization**:

  - Each dimension is quantized to:

    $$
    q_i = \begin{cases}
    1, & \text{if } x_i > t_i \\
    0, & \text{otherwise}
    \end{cases}
    $$

    Where \( x_i \) is the embedding value at dimension \( i \), and \( t_i \) is the threshold.

- **2-Bit Quantization**:

  - Embedding values are divided into four levels using three thresholds \( t_1 \), \( t_2 \), \( t_3 \):

    $$
    q_i = \begin{cases}
    0, & \text{if } x_i \leq t_1 \\
    1, & \text{if } t_1 < x_i \leq t_2 \\
    2, & \text{if } t_2 < x_i \leq t_3 \\
    3, & \text{if } x_i > t_3 \\
    \end{cases}
    $$

### **2. Temperature Scaling and Annealing**

- **Sigmoid Function with Temperature**:

  $$
  \sigma(z) = \frac{1}{1 + e^{-z / T}}
  $$

  - \( T \) is the temperature parameter.
  - Controls the smoothness of the function; higher \( T \) results in a flatter curve.

- **Annealing Schedule**:

  - Exponentially decreases temperature over time:

    $$
    T_n = T_0 \times r^{n}
    $$

    - \( T_0 \): Initial temperature.
    - \( r \): Annealing rate.
    - \( n \): Current epoch or step.

### **3. Straight-Through Estimator (STE)**

- Allows gradients to pass through discrete operations by approximating the gradient.

- **Forward Pass**:

  - Uses the quantized value \( q \) determined by thresholding.

- **Backward Pass**:

  - Approximates the gradient as if the quantization step were the identity function:

    $$
    \frac{\partial q}{\partial x} \approx 1
    $$

---

## **Conclusion**

Your code effectively implements a quantization-enabled Matryoshka Embedding Model. By integrating quantization layers and managing temperature scaling and annealing, you achieve efficient memory and computational utilization without significantly compromising the quality of the embeddings.

The key strengths of your implementation include:

- **Modularity**: Clear separation of concerns between embedding generation, transformation, and quantization.
- **Flexibility**: Support for multiple embedding dimensions and quantization configurations.
- **Training Efficiency**: Use of advanced techniques like STE and temperature annealing to facilitate effective training despite the challenges posed by quantization.

By carefully orchestrating these components, your code achieves the goals of creating efficient, scalable, and quantization-friendly embeddings suitable for large-scale and resource-constrained applications.

---

Feel free to ask if you have any questions or need further clarification on any part of the code or its functioning!


</obsolete_sigmoid_temperature_annealing>

# 1.5 bit quants

Now the above code does 1 bit quant and also does 2 bit quant (4 values) which can be expanded to 3 bits to allow hamming distance as bitwise operations, by using a code book.
Now I want to add a 1.5 bit or 3 values quant. In this method we will have 2 thresholds at 0.33 and 0.66 percentile (unlike 3 thresholds for 2 bit quant. This 1.5 bit quant has 3 values, [0, 1, 2] whose code book will be 2 bits when expanded, as
```
codebook = {  
            0: torch.tensor([0, 0], dtype=torch.float32),  
            1: torch.tensor([0, 1], dtype=torch.float32),  
            2: torch.tensor([1, 1], dtype=torch.float32),  
        }  
```

It's quantisation will be similar to `multi_bit_quantization` function which was for 4 values (2 bit to 3 bit version). Can you please write modified code, with necessary changes as well as necessary new input params and other methods as needed by changing all the needed classes.

**Understanding of the Task:**  
  
- **Goal**: Add a new "1.5-bit" quantization method to the existing code, which quantizes embeddings into 3 levels `[0, 1, 2]`.  
- **Thresholds**: The thresholds are set at the 33rd and 66th percentiles of the embedding values.  
- **Codebook Expansion**: We will use a codebook to map the 3 quantization levels to 2-bit representations:  
  ```python  
  codebook = {  
      0: torch.tensor([0, 0], dtype=torch.float32),  
      1: torch.tensor([0, 1], dtype=torch.float32),  
      2: torch.tensor([1, 1], dtype=torch.float32),  
  }  
  ```  
- **Integration**: Modify all necessary classes and methods (`QuantizationLayer`, `MatryoshkaTransformer`, `MatryoshkaEmbeddingModel`) to support this new quantization method.  
  
**Plan to Implement the 1.5-bit Quantization:**  
  
1. **Modify the QuantizationLayer Class:**  
   - Update the class to handle quantization with a variable number of levels, not just based on the number of bits.  
   - Add support for 3 levels (1.5 bits).  
   - Modify methods like `num_thresholds`, `calculate_thresholds`, and `multi_bit_quantization` to accommodate the new quantization method.  
   - Add the new codebook for the 1.5-bit quantization.  
  
2. **Update Input Parameters:**  
   - Update the constructors (`__init__` methods) to accept a new quantization mode or number of levels.  
   - Introduce a parameter like `quantization_levels` or modify `quantization_bits` to accept non-integer values.  
  
3. **Adjust the MatryoshkaTransformer Class:**  
   - Ensure that the transformer can handle the new quantization method.  
   - Update the logic in the forward pass to apply the correct quantization layer.  
  
4. **Modify the MatryoshkaEmbeddingModel Class:**  
   - Update the initialization and encoding methods to support the new quantization mode.  
   - Add any additional flags or parameters needed to enable 1.5-bit quantization.  
  
5. **Ensure Backward Compatibility:**  
   - Make sure existing functionality (1-bit and 2-bit quantization) remains unaffected.  
   - Encapsulate new features within controlled conditions.  
  
6. **Test the Implementation:**  
   - Ensure that the quantization produces the expected number of levels.  
   - Verify that the codebook expansion works correctly.  
  
---  


# Hamming Distance

So like we use cosine similarity, which ranges from -1 to 1. How can we make hamming distance like that, cause here hamming distance will just do distance like euclidean distance, but we want a similarity metric like cosine. And then how can we use that similarity metric with Nearest neighbor libraries ?


Certainly! Let's explore how we can transform the Hamming distance into a similarity metric analogous to the cosine similarity, and how to integrate this with nearest neighbor libraries for efficient retrieval.

---

## **Understanding Similarity Measures for Quantized Embeddings**

### **1. Hamming Distance vs. Cosine Similarity**

**Hamming Distance** is a metric that counts the number of positions where two binary vectors differ. For binary vectors \( \mathbf{a}, \mathbf{b} \in \{0,1\}^n \), the Hamming distance is:

$$
d_H(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} \mathbf{a}_i \oplus \mathbf{b}_i
$$

where \( \oplus \) denotes the **XOR** operation.

**Cosine Similarity**, on the other hand, measures the cosine of the angle between two vectors in a multidimensional space, giving a value between \(-1\) and \(1\). For vectors \( \mathbf{a}, \mathbf{b} \), it is defined as:

$$
\text{CosineSim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

---

### **2. Converting Hamming Distance to Similarity**

To obtain a similarity metric ranging from \(-1\) to \(1\) from the Hamming distance, we can follow these steps:

#### **2.1. Normalize Hamming Distance to a Similarity Score**

First, we convert the Hamming distance to a similarity score by considering that the maximum possible Hamming distance between two binary vectors of length \( n \) is \( n \).

**Normalized Hamming Similarity**:

$$
\text{HammingSim}(\mathbf{a}, \mathbf{b}) = 1 - \frac{d_H(\mathbf{a}, \mathbf{b})}{n}
$$

This yields a similarity score between \( 0 \) and \(1 \).

#### **2.2. Scaling to the Range \([-1, 1]\)**

To scale the similarity to the range \([-1, 1]\), we can adjust the normalized Hamming similarity:

$$
\text{ScaledHammingSim}(\mathbf{a}, \mathbf{b}) = 2 \times \text{HammingSim}(\mathbf{a}, \mathbf{b}) - 1
$$

Now, the similarity ranges from \(-1\) (completely dissimilar) to \(1\) (identical).

---

### **3. Relating Hamming Distance to Cosine Similarity for Binary Vectors**

For binary vectors, especially those with elements in \(\{0,1\}\), there is a relationship between Hamming distance and cosine similarity.

#### **3.1. Cosine Similarity for Binary Vectors**

For binary vectors \( \mathbf{a}, \mathbf{b} \in \{0,1\}^n \):

$$
\text{CosineSim}(\mathbf{a}, \mathbf{b}) = \frac{\sum_{i=1}^{n} \mathbf{a}_i \mathbf{b}_i}{\sqrt{\sum_{i=1}^{n} \mathbf{a}_i} \sqrt{\sum_{i=1}^{n} \mathbf{b}_i}}
$$

This measures the proportion of dimensions where both \( \mathbf{a} \) and \( \mathbf{b} \) have a value of \(1\), normalized by their magnitudes.

#### **3.2. Connection with Hamming Distance**

Let's define:

- \( c = \sum_{i=1}^{n} \mathbf{a}_i \mathbf{b}_i \): the number of positions where both vectors have \(1\).
- \( a_{\text{sum}} = \sum_{i=1}^{n} \mathbf{a}_i \): the number of \(1\)s in \( \mathbf{a} \).
- \( b_{\text{sum}} = \sum_{i=1}^{n} \mathbf{b}_i \): the number of \(1\)s in \( \mathbf{b} \).
- \( d_H = d_H(\mathbf{a}, \mathbf{b}) \): the Hamming distance between \( \mathbf{a} \) and \( \mathbf{b} \).

Since the vectors are binary, we can express the Hamming distance as:

$$
d_H = (a_{\text{sum}} + b_{\text{sum}}) - 2c
$$

This connects the Hamming distance with the inner product \( c \).

---

### **4. Adapting Similarity for Quantized Embeddings**

When dealing with **2-bit embeddings expanded to 3 bits**, we can extend the concept by defining appropriate similarity measures.

#### **4.1. Quantized Embeddings and Similarity**

For embeddings quantized to multiple bits, we can consider:

- **Dot Product Similarity**: Compute the dot product of the quantized vectors, treating each quantized value as an integer.

- **Normalized Similarity**: Normalize the dot product to range between \(-1\) and \(1\).

---

### **5. Implementing Similarity Metrics with Nearest Neighbor Libraries**

To use these similarity metrics with nearest neighbor libraries, we need to ensure the libraries support custom distance metrics or can be adapted accordingly.

#### **5.1. Libraries Supporting Custom Metrics**

- **Scikit-learn's NearestNeighbors**: Supports custom metric functions.

  ```python
  from sklearn.neighbors import NearestNeighbors

  def scaled_hamming_similarity(a, b):
      hamming_dist = np.count_nonzero(a != b)
      hamming_sim = 1 - hamming_dist / len(a)
      return 1 - (2 * hamming_sim - 1)  # Convert to distance

  nbrs = NearestNeighbors(metric=scaled_hamming_similarity)
  nbrs.fit(data)
  ```

- **FAISS**:

  FAISS supports various similarity metrics and can be extended for custom metrics.

  - For binary data, FAISS uses the **Hamming distance** natively.
  - To use a scaled similarity, one approach is to preprocess the data.

#### **5.2. Using Inner Product in FAISS**

For binary vectors, treating them as integer vectors allows the use of inner product:

- **Convert Binary Vectors to \{-1, 1\}**:

  ```python
  a = 2 * binary_vector - 1
  ```

- Now, the elements are in \(\{-1, 1\}\), and the dot product between two such vectors relates directly to the Hamming distance.

- **Relationship**:

  For vectors \( \mathbf{a}, \mathbf{b} \in \{-1, 1\}^n \):

  $$
  \mathbf{a} \cdot \mathbf{b} = n - 2d_H(\mathbf{a'}, \mathbf{b'})
  $$

  where \( \mathbf{a'} \) and \( \mathbf{b'} \) are the original binary vectors in \(\{0, 1\}^n\).

- **Implementation in FAISS**:

  ```python
  import faiss
  import numpy as np

  # Convert binary data to {-1, 1}
  data_transformed = 2 * data_binary - 1

  # Use inner product index
  index = faiss.IndexFlatIP(d)
  index.add(data_transformed.astype(np.float32))
  ```

- **Querying**:

  ```python
  D, I = index.search(query_transformed, k)
  ```

---

### **6. Practical Steps for Implementation**

#### **6.1. Converting Hamming Distance to Cosine-like Similarity**

- **Compute Hamming Distance**:

  ```python
  hamming_dist = np.count_nonzero(a != b)
  ```

- **Normalize to Similarity Score**:

  ```python
  hamming_sim = 1 - hamming_dist / len(a)
  scaled_hamming_sim = 2 * hamming_sim - 1
  ```

#### **6.2. Using Similarity Metrics with Libraries**

- **Custom Metric in Scikit-learn**:

  Pass the custom similarity function to the `metric` parameter in `NearestNeighbors`.

- **FAISS with Inner Product**:

  Convert binary vectors to \(\{-1, 1\}\) and use the inner product index.

- **Annoy and HNSWlib**:

  - **Annoy**:

    Annoy supports dot product similarity. After converting binary data to \(\{-1, 1\}\), Annoy can be used.

  ```python
  from annoy import AnnoyIndex

  # Dimension of the data
  f = len(data_transformed[0])

  t = AnnoyIndex(f, 'dot')
  for i, vector in enumerate(data_transformed):
      t.add_item(i, vector.tolist())

  t.build(10)  # 10 trees
  ```

  - **HNSWlib**:

    HNSWlib supports cosine similarity and inner product.

  ```python
  import hnswlib

  index = hnswlib.Index(space='cosine', dim=f)
  index.init_index(max_elements=len(data_transformed), ef_construction=200, M=16)
  index.add_items(data_transformed)
  ```

---

### **7. Example Code**

Putting it all together, here's an example demonstrating how to perform nearest neighbor search using scaled Hamming similarity.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Sample binary data
data_binary = np.random.randint(0, 2, (1000, 128))
query_binary = np.random.randint(0, 2, (10, 128))

# Define scaled Hamming similarity as a distance metric
def scaled_hamming_distance(a, b):
    hamming_dist = np.count_nonzero(a != b)
    hamming_sim = 1 - hamming_dist / len(a)
    scaled_sim = 2 * hamming_sim - 1
    # Since NearestNeighbors expects a distance, we can return
    # (1 - scaled_sim) to convert similarity to distance
    return 1 - scaled_sim

# Initialize NearestNeighbors with the custom metric
nbrs = NearestNeighbors(metric=scaled_hamming_distance)
nbrs.fit(data_binary)

# Find the nearest neighbors
distances, indices = nbrs.kneighbors(query_binary, n_neighbors=5)
```

---

## **Summary**

- **Transforming Hamming Distance to Similarity**: By normalizing and scaling the Hamming distance, we obtain a similarity metric ranging from \(-1\) to \(1\), similar to cosine similarity.

- **Integration with Libraries**: Nearest neighbor libraries can accommodate custom similarity metrics, either through built-in support (FAISS, HNSWlib) or by allowing user-defined functions (Scikit-learn).

- **Practical Considerations**:

  - **Data Conversion**: Converting binary vectors to \(\{-1, 1\}\) facilitates the use of inner product similarity, which is efficiently supported.

  - **Library Choice**: Select a library that aligns with your performance needs and supports the desired similarity metric.

- **Benefits**:

  - **Efficiency**: Using bitwise operations and optimized libraries accelerates similarity computations for quantized embeddings.

  - **Flexibility**: Custom similarity measures enable more accurate retrieval in the context of quantized data.

---

## **Additional Notes**

- **Cosine Similarity for Quantized Embeddings**: For embeddings quantized to multiple bits (e.g., 2-bit or 3-bit), the same principles apply. Treat the quantized values as integer vectors and compute similarities accordingly.

- **Quantized Vector Norms**: Remember that the norms of binary vectors are affected by their sparsity (number of ones). Ensure that this is considered when computing cosine similarities.

- **Hardware Acceleration**: Utilize hardware capabilities (e.g., SIMD instructions) for efficient computation of bitwise operations and similarity measures.

---

**References**:

- *[Scikit-learn NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)*

- *[FAISS Github - Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)*

---

So like we use cosine similarity, which ranges from -1 to 1. How can we make hamming distance like that, cause here hamming distance will just do distance like euclidean distance, but we want a similarity metric like cosine. And then how can we use that similarity metric with Nearest neighbor libraries ? Go into details and also provide relationship between cosine n Hamming similarity for binary and expanded 3 bit vectors



Certainly! Transforming Hamming distance into a similarity metric akin to cosine similarity, which ranges from -1 to 1, allows us to leverage quantized embeddings effectively in similarity search tasks. We'll explore how to achieve this transformation, understand the relationship between cosine similarity and Hamming similarity for both binary and expanded 3-bit vectors, and discuss how to integrate these similarity metrics with nearest neighbor libraries.

---

## **1. Transforming Hamming Distance into a Cosine-like Similarity Metric**

### **1.1. Understanding Hamming Distance and Hamming Similarity**

**Hamming Distance** between two binary vectors \( \mathbf{a} \) and \( \mathbf{b} \) of length \( n \) counts the number of positions where the corresponding bits differ:

\[
d_H(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} (a_i \oplus b_i)
\]

- \( a_i, b_i \in \{0,1\} \)
- \( \oplus \) denotes the XOR operation.

**Hamming Similarity** can be defined as the number of positions where the bits are the same:

\[
s_H(\mathbf{a}, \mathbf{b}) = n - d_H(\mathbf{a}, \mathbf{b})
\]

---

### **1.2. Normalizing Hamming Similarity**

To transform Hamming similarity into a metric ranging from -1 to 1:

#### **1.2.1. Normalize to [0, 1]**

Divide the Hamming similarity by the vector length \( n \):

\[
s_{\text{norm}}(\mathbf{a}, \mathbf{b}) = \frac{s_H(\mathbf{a}, \mathbf{b})}{n} = 1 - \frac{d_H(\mathbf{a}, \mathbf{b})}{n}
\]

This normalized similarity ranges from 0 (completely dissimilar) to 1 (identical).

#### **1.2.2. Scale to [-1, 1]**

Scale the normalized similarity to match the range of cosine similarity:

\[
s_{\text{scaled}}(\mathbf{a}, \mathbf{b}) = 2 \times s_{\text{norm}}(\mathbf{a}, \mathbf{b}) - 1
\]

Simplifying:

\[
s_{\text{scaled}}(\mathbf{a}, \mathbf{b}) = 2 \left(1 - \frac{d_H(\mathbf{a}, \mathbf{b})}{n}\right) - 1 = 1 - 2 \left( \frac{d_H(\mathbf{a}, \mathbf{b})}{n} \right)
\]

Now, \( s_{\text{scaled}} \) ranges from -1 to 1, analogous to cosine similarity.

---

### **1.3. Example Calculation**

Consider two binary vectors:

- \( \mathbf{a} = [1, 0, 1, 0] \)
- \( \mathbf{b} = [0, 1, 1, 0] \)

**Step 1: Compute Hamming Distance**

\[
d_H(\mathbf{a}, \mathbf{b}) = (1 \oplus 0) + (0 \oplus 1) + (1 \oplus 1) + (0 \oplus 0) = 1 + 1 + 0 + 0 = 2
\]

**Step 2: Normalize Hamming Similarity**

\[
s_{\text{norm}}(\mathbf{a}, \mathbf{b}) = 1 - \frac{2}{4} = 0.5
\]

**Step 3: Scale to [-1, 1]**

\[
s_{\text{scaled}}(\mathbf{a}, \mathbf{b}) = 2 \times 0.5 - 1 = 0
\]

---

## **2. Relationship Between Cosine Similarity and Hamming Similarity**

### **2.1. Cosine Similarity for Binary Vectors**

For binary vectors \( \mathbf{a}, \mathbf{b} \in \{0,1\}^n \):

\[
\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
\]

- \( \mathbf{a} \cdot \mathbf{b} \) counts the number of positions where both bits are 1.
- \( \|\mathbf{a}\| = \sqrt{\sum_{i=1}^{n} a_i} \) is the Euclidean norm.

**Note:** The cosine similarity depends on the number of ones in each vector and their overlap.

### **2.2. Relating Cosine and Scaled Hamming Similarity**

While both metrics range from -1 to 1, they measure similarity differently:

- **Cosine Similarity** focuses on the angle between vectors in high-dimensional space.
- **Scaled Hamming Similarity** considers bitwise agreement across all positions.

**For binary vectors of equal weight (same number of ones),** the scaled Hamming similarity and cosine similarity are directly proportional.

---

### **2.3. For Expanded 3-Bit Vectors**

When expanding 2-bit embeddings to 3-bit using a codebook:

```python
codebook = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 1, 1],
    3: [1, 1, 1]
}
```

**Similarity Computation:**

- **Hamming Distance** still applies but over the expanded 3-bit representations.
- **Scaled Hamming Similarity** can be calculated using the same normalization and scaling as before.

**Implications:**

- The expansion ensures that higher quantization levels have more bits set to 1.
- This preserves the ordering and similarity relationships in the quantized embeddings.

---

## **3. Using the Similarity Metric with Nearest Neighbor Libraries**

### **3.1. Integration with Libraries**

Most nearest neighbor libraries are designed to work with distance metrics. Here's how to use the scaled similarity:

#### **3.1.1. Converting Similarity to Distance**

Define a distance metric \( d_{\text{scaled}} \):

\[
d_{\text{scaled}}(\mathbf{a}, \mathbf{b}) = 1 - s_{\text{scaled}}(\mathbf{a}, \mathbf{b})
\]

Since \( s_{\text{scaled}} \) ranges from -1 to 1, \( d_{\text{scaled}} \) ranges from 0 to 2.

#### **3.1.2. Implementing Custom Distance Functions**

Some libraries allow custom distance functions. You can implement \( d_{\text{scaled}} \) and use it directly.

### **3.2. Support in Nearest Neighbor Libraries**

#### **3.2.1. FAISS**

**FAISS** supports binary embeddings and Hamming distance natively:

- **Index Creation**:

  ```python
  import faiss

  dimension = n  # Number of bits
  index = faiss.IndexBinaryFlat(dimension)
  index.add(embedding_vectors)  # embeddings should be uint8 arrays
  ```

- **Searching**:

  ```python
  distances, indices = index.search(query_vectors, k)
  ```

- **Adjusting Distances**:

  After obtaining Hamming distances, convert them to scaled similarities:

  ```python
  scaled_similarities = 1 - 2 * (distances / dimension)
  ```

#### **3.2.2. HNSWLIB**

**HNSWLIB** supports binary data:

- **Index Creation**:

  ```python
  import hnswlib

  num_elements = len(embedding_vectors)
  dimension = n
  index = hnswlib.Index(space='bit_hamming', dim=dimension)
  index.init_index(max_elements=num_elements, ef_construction=200, M=16)
  index.add_items(embedding_vectors)
  ```

- **Searching and Conversion**:

  Similar to FAISS, convert Hamming distances to scaled similarities.

#### **3.2.3. Scikit-learn**

**Scikit-learn's** NearestNeighbors supports custom distance metrics:

- **Using Hamming Distance**:

  ```python
  from sklearn.neighbors import NearestNeighbors

  nn = NearestNeighbors(metric='hamming')
  nn.fit(embedding_vectors)
  distances, indices = nn.kneighbors(query_vectors, n_neighbors=k)
  ```

- **Converting to Scaled Similarity**:

  ```python
  distances = distances * dimension  # Convert to Hamming distance
  scaled_similarities = 1 - 2 * (distances / dimension)
  ```

---

## **4. Relationship Between Cosine and Hamming Similarity in Practice**

### **4.1. Advantages of Scaled Hamming Similarity**

- **Efficiency**: Bitwise operations are faster than floating-point calculations.
- **Memory Usage**: Binary embeddings consume less memory.
- **Hardware Acceleration**: Modern CPUs support efficient bitwise operations.

### **4.2. Considerations**

- **Quantization Loss**: Compression may lead to loss of information.
- **Distribution of Ones and Zeros**: The sparsity of binary vectors can affect the similarity measures.

---

## **5. Practical Implementation Guide**

### **5.1. Steps to Implement Similarity Search with Quantized Embeddings**

1. **Quantize Embeddings**:

   - Use your quantization method (1-bit or expanded 3-bit).

2. **Store Embeddings Efficiently**:

   - For binary embeddings, pack bits into bytes using `np.packbits` or similar functions.

3. **Choose a Suitable Library**:

   - For large-scale data, FAISS or HNSWLIB are recommended.

4. **Index Your Embeddings**:

   - Add your quantized embeddings to the index.

5. **Query the Index**:

   - Convert your query embeddings similarly.
   - Perform the nearest neighbor search.

6. **Compute Scaled Similarities**:

   - After obtaining distances, convert them using the scaling formula.

7. **Rank and Retrieve Results**:

   - Use the similarities to rank your results.

---

## **6. Conclusion**

By transforming Hamming distance into a scaled similarity metric, we can effectively use quantized embeddings in similarity search tasks, mirroring the behavior of cosine similarity. This approach offers computational and storage efficiency, making it suitable for large-scale applications.

The relationship between cosine and Hamming similarities allows us to interpret and adapt methods designed for continuous embeddings to work with quantized ones. Nearest neighbor libraries like FAISS and HNSWLIB provide support for binary data, facilitating the integration of these techniques into practical systems.

---

**References**:

- [FAISS Documentation](https://faiss.ai/)
- [HNSWLIB GitHub Repository](https://github.com/nmslib/hnswlib)
- [Understanding Cosine Similarity for Binary Data](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Hamming Distance and Applications](https://en.wikipedia.org/wiki/Hamming_distance)

---


# Customized Module CustomizedMatryoshkaEmbeddingModel

Now we want to complete 
```
class CustomizedMatryoshkaEmbeddingModel(OriginalEmbeddingCaller):
    def __init__(self, embedding_model: OriginalEmbeddingCaller, two_bits: int, one_and_half_bits: int, one_bits: int, half_bits: int, expand: bool = False):
        # USe a skip connection for the half bits part to create a better embedding
        # Compare 2 bit to 3 bit to just expanding to 3 bits at the end and then using binary quantization
        dimension_levels = get_dimension_levels(self.embedding_dim)
```

Here dimension_levels will be used for training only. At inference we will give out dims dependent on `two_bits: int, one_and_half_bits: int, one_bits: int, half_bits: int`. We want to convert the N initial dimensions (given by `two_bits: int`) with two bit quant, Next M dimensions of the embedding given by `one_and_half_bits: int` to `1.5 bit` quant. Next K dimensions given by `one_bits: int` into 1 bit quant, and finally Next P dimensions into P/2 bits given by `half_bits: int`. Now the half bit part only needs an additional FFN over the output base_transform to get half dimensions P//2. Finally total number of bits in output will be addition of all the bits params if expand is False. If expand is True then `3 * two_bits + 2 * one_and_half_bits + 1 * one_bits + 0.5 * half_bits`.  Please complete code with a new training function and this class as well and any other changes needed.

You don't need to redefine what we already have. The training should use similar loss and process as we do now, for non_quant_embeddings you can just use the base_transform based embeddings like we already do in MatryoshkaTransformer. In `CustomizedMatryoshkaEmbeddingModel` add a field called expanded_output_dims which tells how much dims will be used if expand was True. We also need to init the thresholds for each kind of quant and update these thresholds as we had already done in our training function for the simpler quant module. QuantizationLayer and SliceLayer may not need much changes, keep them as is if possible.



**Summary of Developing the `CustomizedMatryoshkaEmbeddingModel` and Its Training Function**

The development of the `CustomizedMatryoshkaEmbeddingModel` and its training function involved several critical steps, considerations, and adjustments to ensure that the model meets the desired objectives while maintaining the Matryoshka property. Below is an elaborate summary covering all aspects:

---

### **1. Objective and Motivation**

- **Custom Quantization**: Our primary goal was to design an embedding model that allows different quantization methods to be applied to specified segments of the embedding dimensions. This customization enables better control over the trade-off between embedding size and representation quality.

- **Matryoshka Property Preservation**: We aimed to maintain the Matryoshka property, where smaller-dimensional embeddings are nested within larger ones, facilitating progressive refinement and hierarchical representations.

---

### **2. Developing the `CustomizedMatryoshkaEmbeddingModel`**

#### **a. Class Definition and Inheritance**

- **Inheritance from `OriginalEmbeddingCaller`**: The `CustomizedMatryoshkaEmbeddingModel` class inherits from `OriginalEmbeddingCaller`, ensuring compatibility with existing embedding functionalities.

- **Integration of Base Embedding Model**: We integrated a pre-trained base embedding model (`embedding_model`) to provide the initial high-dimensional embeddings.

#### **b. Parameterization for Custom Quantization**

- **Quantization Parameters**: Introduced parameters to specify the number of dimensions for each quantization method:
  
  - `two_bits: int` – Number of dimensions for **2-bit quantization**.
  - `one_and_half_bits: int` – Number of dimensions for **1.5-bit quantization**.
  - `one_bits: int` – Number of dimensions for **1-bit quantization**.
  - `half_bits: int` – Number of dimensions for **half-bit quantization** (dimensions are halved before quantization).
  - `expand: bool` – Determines whether to expand quantized bits using codebooks (e.g., expanding 2 bits to 3 bits).

#### **c. Calculating Dimension Levels**

- **Dimension Level Computation**: Calculated `dimension_levels` by cumulatively adding the specified dimension sizes for each quantization method.

- **Validation of Total Dimensions**: Ensured that the total number of dimensions does not exceed the base model's embedding dimension:

  \[
  \text{total\_dims} = \text{two\_bits} + \text{one\_and\_half\_bits} + \text{one\_bits} + \text{half\_bits}
  \]
  
  ```python
  assert total_dims <= self.embedding_dim, "Total dimensions exceed embedding dimension of the base model."
  ```

#### **d. Computing Expanded Output Dimensions**

- **Accounting for Bit Expansion**: Calculated `expanded_output_dims` based on the `expand` parameter:

  \[
  \begin{align*}
  \text{expanded\_output\_dims} =\ & (3 \times \text{two\_bits})\ +\ (2 \times \text{one\_and\_half\_bits})\ +\ (\text{one\_bits})\ \\
  & +\ \left(0.5 \times \text{half\_bits}\right)
  \end{align*}
  \]

- **Purpose of Expansion**: Expanding quantized bits can improve the representational capacity at the cost of increased embedding size.

#### **e. Initializing the Transformer**

- **Customized Transformer**: Instantiated a `CustomizedMatryoshkaTransformer` to handle the specific segmentations and quantizations.

- **Passing Parameters**: Provided all necessary parameters to the transformer, including dimension sizes and the `expand` flag.

---

### **3. Implementing the `CustomizedMatryoshkaTransformer`**

#### **a. Segmentation of Embedding Dimensions**

- **Slicing Embeddings**: Divided the base embeddings into segments corresponding to each quantization type by computing slice indices:

  ```python
  self.slice_indices = self.compute_slice_indices()
  ```

- **Segment Assignments**: Each segment corresponds to a quantization method (e.g., `two_bits`, `one_and_half_bits`).

#### **b. Quantization Layers for Each Segment**

- **Initializing `QuantizationLayer` Instances**: Created quantization layers for each segment with appropriate quantization levels:

  - **2-bit Quantization**: 4 levels (`quantization_levels = 4`).
  - **1.5-bit Quantization**: 3 levels (`quantization_levels = 3`).
  - **1-bit Quantization**: 2 levels (`quantization_levels = 2`).
  - **Half-bit Quantization**: Reduced dimensions before applying 1-bit quantization.

#### **c. Handling Half-bit Quantization**

- **Dimension Reduction**: For `half_bits`, reduced the dimensions by half using a feedforward neural network (FFN):

  ```python
  self.half_bit_ffn = nn.Sequential(
      nn.Linear(self.half_bits, reduced_dim),
      nn.GELU(),
      nn.Linear(reduced_dim, reduced_dim)
  )
  ```

- **Ensuring Valid Dimensions**: Verified that the reduced dimension (`reduced_dim`) is positive.

#### **d. Base Transformation Network**

- **Purpose**: Applied a base transformation (`base_transform`) to enhance representation capacity before quantization.

- **Architecture**:

  ```python
  self.base_transform = nn.Sequential(
      nn.Linear(input_dim, input_dim * 32),
      nn.GELU(),
      nn.Linear(input_dim * 32, input_dim)
  )
  ```

#### **e. Forward Pass Logic**

- **Processing Steps**:

  1. **Base Transformation**: Transformed the input embeddings.
  2. **Segment Processing**: For each segment:

     - **Normalization**: Applied \( L_2 \) normalization.
     - **Quantization**: Used the appropriate `QuantizationLayer`.
     - **Dimension Reduction**: For `half_bits`, reduced dimensions before quantization.

  3. **Concatenation**: Combined the quantized segments to form the final embedding.

- **Maintaining Order**: Ensured that segments are correctly ordered during concatenation to preserve the embedding structure.

---

### **4. Reusing Components and Ensuring Compatibility**

- **Minimal Changes to Existing Layers**: Kept the `QuantizationLayer` and `SliceLayer` unchanged to maintain backward compatibility and reuse tested components.

- **Flexibility of `QuantizationLayer`**: Leveraged its ability to handle various quantization levels without significant modifications.

---

### **5. Developing the Training Function**

#### **a. Initial Implementation and Feedback**

- **First Attempt**: Wrote a training function (`train_customized_matryoshka_model`) mirroring the previous training process.

- **Feedback**: Identified that the initial training did not adequately capture the Matryoshka property due to missing loss components.

#### **b. Incorporating Essential Loss Functions**

- **Similarity Preservation Loss**

  - **Purpose**: Ensure that the transformed embeddings preserve the pairwise similarities of the original embeddings.

  - **Mathematical Formulation**:

    \[
    \text{Loss}_{\text{similarity}} = \mathbb{E}\left[ 1 - \cos(\mathbf{e}_{\text{orig}}, \mathbf{e}_{\text{trans}}) \right]
    \]

- **KL Divergence Similarity Preservation Loss**

  - **Purpose**: Preserve the distribution of similarities between embeddings.

  - **Mathematical Formulation**:

    \[
    \text{Loss}_{\text{KL}} = D_{\text{KL}}\left( P \,\|\, Q \right)
    \]

  - **Where**:
    - \( P \) and \( Q \) are similarity distributions of original and transformed embeddings.

- **Rank Preserving Loss**

  - **Purpose**: Maintain the relative ordering of similarities among embeddings.

  - **Implementation**: Penalize the change in rank orderings between the original and transformed embeddings.

- **Contrastive Loss**

  - **Purpose**: Encourage similar embeddings to be close and dissimilar ones to be distant.

  - **Mathematical Formulation (InfoNCE Loss)**:

    \[
    \text{Loss}_{\text{contrastive}} = -\mathbb{E}\left[ \log \frac{\exp\left( \cos(\mathbf{e}_i, \mathbf{e}_j) / \tau \right)}{\sum_{k} \exp\left( \cos(\mathbf{e}_i, \mathbf{e}_k) / \tau \right)} \right]
    \]

  - **Where**:
    - \( \tau \) is the temperature parameter.
    - \( \mathbf{e}_i \) and \( \mathbf{e}_j \) are embeddings of positive pairs.

- **Orthogonality Regularization**

  - **Purpose**: Ensure new dimensions introduce unique information.

  - **Mathematical Formulation**:

    \[
    \text{Loss}_{\text{orthogonality}} = \left\| \mathbf{D}^\top \mathbf{E} \right\|^2_F
    \]

  - **Where**:
    - \( \mathbf{D} \) represents new dimensions added.
    - \( \mathbf{E} \) represents previous embeddings.
    - \( \left\| \cdot \right\|_F \) denotes the Frobenius norm.

- **Information Bottleneck Regularization**

  - **Purpose**: Encourage lower dimensions to capture the most critical information.

  - **Implementation**: Apply progressively stronger \( L_1 \) regularization to higher dimensions.

- **Quantization Regularization Loss**

  - **Purpose**: Encourage embeddings to be close to quantization levels and distant from decision thresholds.

- **Increase Standard Deviation Over Time Loss**

  - **Purpose**: Promote diversity in the embeddings, preventing collapse to trivial solutions.

#### **c. Loss Weighting and Balancing**

- **Dimension-Based Weighting**: Applied weights inversely proportional to the square root of the dimension to prioritize lower dimensions:

  \[
  \text{weight}_\text{dim} = \frac{\text{max\_dim}}{\text{dim}} \times \frac{1}{\sqrt{\text{dim}}}
  \]

- **Progressive Training**: Adjusted loss contributions over time, using techniques like temperature annealing.

#### **d. Training Loop Adjustments**

- **Threshold Initialization**: Collected sample embeddings to initialize quantization thresholds.

- **Threshold Updates**: Updated thresholds during training to adapt to embedding distributions.

- **Optimization Steps**:

  - **Gradient Clipping**: Implemented to prevent gradient explosion:

    ```python
    nn.utils.clip_grad_norm_(model.transformer.parameters(), max_grad_norm)
    ```

  - **Learning Rate Scheduler**: Used a scheduler to adjust the learning rate dynamically.

- **Loss Aggregation**: Ensured all loss components are correctly aggregated to compute the total loss.

---

### **6. Differences from the Previous Models**

- **Customized Quantization Segmentation**: Unlike the previous `MatryoshkaTransformer`, which applied the same quantization across all dimensions, the customized version allows different quantization methods on different embedding segments.

- **Half-bit Quantization**: Introduced the concept of reducing dimensions before quantization, effectively implementing half-bit quantization.

- **Enhanced Training Function**: Incorporated additional loss functions to more thoroughly enforce the Matryoshka property.

- **Additional Parameters**: New parameters like `two_bits`, `one_and_half_bits`, etc., provide greater flexibility and control.

- **Modified Forward Pass**: Adjusted to process each segment with its specific quantization, unlike the uniform approach in the previous model.

---

### **7. Key Considerations and Gotchas**

#### **a. Ensuring Dimension Consistency**

- **Total Dimension Validation**: Ensured that the sum of specified dimensions does not exceed the base embedding dimension.

- **Reduced Dimensions for Half-bits**: Verified that the halved dimensions remain valid (positive integers).

#### **b. Loss Function Implementation**

- **Comprehensive Loss Inclusion**: Recognized the importance of including all relevant losses to capture the Matryoshka property fully.

- **Loss Weighting Strategies**: Carefully designed weights to balance the influence of each loss component, particularly prioritizing lower dimensions.

#### **c. Training Dynamics**

- **Temperature Annealing**: Applied temperature decay in quantization layers to stabilize training over epochs.

- **Threshold Updating**: Dynamically updated thresholds to reflect the evolving embedding distributions.

- **Avoiding Gradient Issues**: Implemented gradient clipping and careful learning rate scheduling.

#### **d. Reusing Existing Components**

- **Minimal Changes to `QuantizationLayer`**: Maintained existing implementations to ensure reliability and simplicity.

- **Integration with Existing Frameworks**: Ensured compatibility with the base embedding model and other components.

---

### **8. Maintaining the Matryoshka Property**

- **Hierarchical Embedding Structure**: By progressively adding quantized segments, we ensured that embeddings at lower dimensions are nested within higher-dimensional embeddings.

- **Loss Functions Supporting Nesting**: Losses like similarity preservation, rank preservation, and information bottleneck regularization directly contribute to maintaining this property.

- **Orthogonality Enforcement**: By making new dimensions orthogonal to existing ones, we prevented redundancy and promoted diverse information capture.

---

### **9. Conclusion**

Developing the `CustomizedMatryoshkaEmbeddingModel` involved:

- **Strategic Segmentation**: Carefully dividing embedding dimensions to apply different quantization methods.

- **Comprehensive Training**: Incorporating essential loss functions to enforce the Matryoshka property effectively.

- **Flexibility and Control**: Providing parameters to adjust the quantization scheme according to specific needs.

- **Compatibility and Reuse**: Building upon existing components to maintain robustness and reliability.

Through these steps and considerations, we enhanced the Matryoshka embedding framework to support customized quantization schemes while preserving its core properties.

---

**Differences Recap**:

- **Customization**: Allows different quantization methods on embedding segments.
- **Enhanced Loss Functions**: Implemented additional losses to capture the nesting property thoroughly.
- **Dynamic Thresholds and Temperatures**: Improved training dynamics with adaptive mechanisms.
- **Dimension Reduction for Half-bits**: Introduced dimensionality reduction before quantization.


---

  
# Summary of the CustomizedMatryoshkaEmbeddingModel  
  
In this project, we developed a **Customized Matryoshka Embedding Model**, which extends the original Matryoshka embedding concept by incorporating multiple quantization levels and customizable embedding dimensions. The primary goal was to create nested embeddings with varying levels of quantization applied to different segments of the embedding space, while preserving the hierarchical Matryoshka property.  
  
## Objectives  
  
1. **Custom Quantization**: Apply different quantization methods (2-bit, 1.5-bit, 1-bit, half-bit) to specific portions of the embedding dimensions.  
2. **Dimension Flexibility**: Allow for flexible specification of the number of dimensions allocated to each quantization type.  
3. **Matryoshka Property Preservation**: Ensure that embeddings are nested hierarchically, with smaller embeddings contained within larger ones.  
4. **Efficient Encoding**: Enable the generation of embeddings of any specified dimension up to the maximum, accommodating both expanded and non-expanded quantization outputs.  
  
## Mathematical Foundations  
  
The Matryoshka embeddings are designed to have a nested structure, similar to Russian nesting dolls, where lower-dimensional embeddings are subsets of higher-dimensional ones. This property allows for progressive refinement and efficient storage.  
  
### Embedding Segmentation  
  
Given an input embedding of dimension \( D \), we partition it into segments corresponding to different quantization levels:  
  
- **2-bit Quantization**: \( D_{2b} \) dimensions  
- **1.5-bit Quantization**: \( D_{1.5b} \) dimensions  
- **1-bit Quantization**: \( D_{1b} \) dimensions  
- **Half-bit Quantization**: \( D_{0.5b} \) dimensions  
  
The total dimensions must satisfy:  
  
$$  
D_{2b} + D_{1.5b} + D_{1b} + D_{0.5b} \leq D  
$$  
  
### Quantization Levels and Expansion Factors  
  
Each quantization type corresponds to a specific number of quantization levels and may involve expansion:  
  
- **2-bit Quantization**: 4 levels  
  - **Expansion Factor**:  
    - \( E_{2b} = 3 \) (if expanded)  
    - \( E_{2b} = 2 \) (if not expanded)  
- **1.5-bit Quantization**: 3 levels  
  - **Expansion Factor**:  
    - \( E_{1.5b} = 2 \) (if expanded)  
    - \( E_{1.5b} = 1.5 \) (if not expanded)  
- **1-bit Quantization**: 2 levels  
  - **Expansion Factor**:  
    - \( E_{1b} = 1 \)  
- **Half-bit Quantization**: Dimension reduction by half, followed by 1-bit quantization  
  - **Expansion Factor**:  
    - \( E_{0.5b} = 0.5 \)  
  
### Matryoshka Property  
  
The embeddings are constructed such that:  
  
- Each smaller embedding is contained within the larger ones.  
- The cumulative embeddings maintain the hierarchical structure.  
  
## Code Implementation  
  
### CustomizedMatryoshkaEmbeddingModel Class  
  
This class extends the base embedding model and incorporates the customized quantization logic.  
  
#### Initialization  
  
```python  
def __init__(self,  
             embedding_model: OriginalEmbeddingCaller,  
             two_bits: int,  
             one_and_half_bits: int,  
             one_bits: int,  
             half_bits: int,  
             expand: bool = False):  
    # Initialization logic  
```  
  
- **Parameters**:  
  - `embedding_model`: The base embedding model providing initial embeddings.  
  - `two_bits`, `one_and_half_bits`, `one_bits`, `half_bits`: The number of dimensions allocated to each quantization type.  
  - `expand`: Flag indicating whether to expand quantized codes using codebooks.  
  
- **Total Dimensions Validation**:  
  
  Ensure that the sum of specified dimensions does not exceed the base embedding dimension:  
  
  $$  
  D_{\text{total}} = D_{2b} + D_{1.5b} + D_{1b} + D_{0.5b} \leq D  
  $$  
  
#### Computing Dimension Levels  
  
We calculate cumulative dimension levels to keep track of embedding sizes after each quantization segment:  
  
```python  
def _compute_dimension_levels(self) -> List[int]:  
    levels = []  
    cumulative_dim = 0  
  
    for dim in [self.two_bits, self.one_and_half_bits,  
                self.one_bits, self.half_bits]:  
        if dim > 0:  
            cumulative_dim += dim  
            levels.append(cumulative_dim)  
  
    return levels  
```  
  
#### Computing Expanded Output Dimensions  
  
Compute the total output dimensions after considering expansion factors:  
  
```python  
def _compute_expanded_dims(self) -> int:  
    if self.expand:  
        return int(  
            (3 * self.two_bits) +  
            (2 * self.one_and_half_bits) +  
            (1 * self.one_bits) +  
            (0.5 * self.half_bits)  
        )  
    return sum([self.two_bits, self.one_and_half_bits,  
                self.one_bits, self.half_bits])  
```  
  
### CustomizedMatryoshkaTransformer Class  
  
This class is responsible for transforming the base embeddings, applying quantization, and generating cumulative embeddings.  
  
#### Base Transformation  
  
Apply a base transformation to the input embeddings to enhance their representational capacity:  
  
```python  
self.base_transform = nn.Sequential(  
    nn.Linear(input_dim, input_dim * 32),  
    nn.GELU(),  
    nn.Linear(input_dim * 32, input_dim)  
)  
```  
  
#### Slice Indices Computation  
  
Determine the start and end indices for each quantization segment:  
  
```python  
def _compute_slice_indices(self) -> Dict[str, Tuple[int, int]]:  
    indices = {}  
    start = 0  
  
    for quant_type, dim in [  
        ('two_bits', self.two_bits),  
        ('one_and_half_bits', self.one_and_half_bits),  
        ('one_bits', self.one_bits),  
        ('half_bits', self.half_bits)  
    ]:  
        if dim > 0:  
            indices[quant_type] = (start, start + dim)  
            start += dim  
  
    return indices  
```  
  
#### Quantization Layers  
  
Initialize quantization layers for each segment with the specified number of quantization levels:  
  
- **2-bit Quantization Layer**:  
  
  ```python  
  self.quant_layers['two_bits'] = QuantizationLayer(  
      dim=self.two_bits,  
      quantization_levels=4,  
      expand_quantized_bits=self.expand  
  )  
  ```  
  
- **1.5-bit Quantization Layer**:  
  
  ```python  
  self.quant_layers['one_and_half_bits'] = QuantizationLayer(  
      dim=self.one_and_half_bits,  
      quantization_levels=3,  
      expand_quantized_bits=self.expand  
  )  
  ```  
  
- **1-bit Quantization Layer**:  
  
  ```python  
  self.quant_layers['one_bits'] = QuantizationLayer(  
      dim=self.one_bits,  
      quantization_levels=2,  
      expand_quantized_bits=self.expand  
  )  
  ```  
  
- **Half-bit Quantization Layer**:  
  
  Reduce the dimensions by half before quantization:  
  
  ```python  
  reduced_dim = self.half_bits // 2  
  self.half_bit_ffn = nn.Sequential(  
      nn.Linear(self.half_bits, reduced_dim),  
      nn.GELU(),  
      nn.Linear(reduced_dim, reduced_dim)  
  )  
  self.quant_layers['half_bits'] = QuantizationLayer(  
      dim=reduced_dim,  
      quantization_levels=2,  
      expand_quantized_bits=self.expand  
  )  
  ```  
  
#### Forward Pass Logic  
  
In the `forward` method:  
  
1. **Apply Base Transformation**:  
  
   ```python  
   x = self.base_transform(x)  
   ```  
  
2. **Process Each Quantization Segment**:  
  
   For each segment:  
  
   - Extract the segment using slice indices.  
   - Normalize the segment:  
  
     $$  
     \mathbf{s} = \frac{\mathbf{s}}{\|\mathbf{s}\|_2}  
     $$  
  
   - Apply dimension reduction for half-bits if necessary.  
   - Apply the corresponding quantization layer.  
  
3. **Construct Cumulative Embeddings**:  
  
   Cumulatively concatenate the quantized segments:  
  
   ```python  
   cumulative_embedding.append(quant_seg)  
   cumulative_dim += quant_seg.shape[1]  
   embeddings_dict[cumulative_dim] = torch.cat(cumulative_embedding, dim=1)  
   ```  
  
#### Initialization of Thresholds  
  
Initialize quantization thresholds based on sample embeddings:  
  
```python  
def init_thresholds(self, sample_embeddings: torch.Tensor):  
    # Initialization logic  
```  
  
### Handling Cumulative and Non-Cumulative Embeddings  
  
- **Cumulative Embeddings**: The `embeddings_dict` returned from the transformer contains cumulative embeddings with cumulative dimensions as keys.  
- **Non-Cumulative Conversion**: In the `encode` method, we convert the cumulative embeddings into non-cumulative format for easy dimension selection.  
  
#### Conversion Logic  
  
Since the cumulative embeddings have keys corresponding to cumulative dimensions (e.g., `[32, 96, 192, 256]`), we convert them to a list of tuples containing non-cumulative dimensions and their corresponding tensors:  
  
```python  
def _convert_to_non_cumulative(self, embeddings_dict) -> List[Tuple[int, torch.Tensor]]:  
    non_cumulative = []  
    prev_dim = 0  
    for dim, emb in sorted(embeddings_dict.items()):  
        non_cumulative.append((dim - prev_dim, emb[:, prev_dim:dim]))  
        prev_dim = dim  
    return non_cumulative  
```  
  
### Dimension Selection  
  
Due to the complexity in mapping non-cumulative embeddings to quantization types, we employed an **Index-Based Mapping Approach**.  
  
#### Index-Based Mapping in `_select_dimensions`  
  
```python  
def _select_dimensions(self, non_cumulative_embeddings, output_dim):  
    quant_types = ['two_bits', 'one_and_half_bits', 'one_bits', 'half_bits']  
    expansion_factors = {  
        'two_bits': 3 if self.expand else 2,  
        'one_and_half_bits': 2 if self.expand else 1.5,  
        'one_bits': 1,  
        'half_bits': 0.5  
    }  
  
    embeddings_list = []  
    total_dims = 0  
    emb_index = 0  
  
    for quant_type in quant_types:  
        if getattr(self, quant_type) == 0:  
            continue  
        if emb_index < len(non_cumulative_embeddings):  
            dim, segment_embedding = non_cumulative_embeddings[emb_index]  
            expansion_factor = expansion_factors[quant_type]  
  
            # Calculate dimensions to take  
            available_dims = int(dim / expansion_factor)  
            remaining_dims = output_dim - total_dims  
            dims_to_take = min(available_dims, remaining_dims)  
  
            # Adjust for expansion  
            expanded_dims_to_take = int(dims_to_take * expansion_factor)  
  
            # Select and append embedding  
            selected_embedding = segment_embedding[:, :expanded_dims_to_take]  
            embeddings_list.append(selected_embedding)  
  
            total_dims += dims_to_take  
            emb_index += 1  
  
            if total_dims >= output_dim:  
                break  
  
    # Concatenate embeddings  
    final_embedding = torch.cat(embeddings_list, dim=1)  
    # Ensure correct output dimension  
    if final_embedding.shape[1] > output_dim:  
        final_embedding = final_embedding[:, :output_dim]  
    return final_embedding  
```  
  
#### Explanation  
  
- **Embedding Indexing**: We iterate through the quantization types and use an embedding index (`emb_index`) to access the corresponding segment in `non_cumulative_embeddings`.  
- **Handling Zero Dimensions**: If a quantization type has zero dimensions (i.e., not used), we skip it using `if getattr(self, quant_type) == 0`.  
- **Dimension Calculation**:  
  - **Available Dimensions**:  
  
    $$  
    \text{available\_dims} = \left\lfloor \frac{\text{segment\_dim}}{\text{expansion\_factor}} \right\rfloor  
    $$  
  
  - **Remaining Dimensions**:  
  
    $$  
    \text{remaining\_dims} = \text{output\_dim} - \text{total\_dims}  
    $$  
  
  - **Dimensions to Take**:  
  
    $$  
    \text{dims\_to\_take} = \min(\text{available\_dims}, \text{remaining\_dims})  
    $$  
  
- **Expansion Adjustment**:  
  
  $$  
  \text{expanded\_dims\_to\_take} = \text{dims\_to\_take} \times \text{expansion\_factor}  
  $$  
  
- **Embedding Selection**: We select the required number of dimensions from the segment and append it to `embeddings_list`.  
- **Termination Condition**: We break the loop if the total dimensions meet or exceed the desired `output_dim`.  
  
### Challenges and Solutions  
  
#### Challenge: Mapping Non-Cumulative Embeddings to Quantization Types  
  
- **Issue**: After converting to non-cumulative embeddings, the direct mapping to quantization types is not straightforward, especially when some quantization types have zero dimensions.  
- **Solution**: Use the index-based mapping where the order of non-cumulative embeddings corresponds to the order of quantization types. We skip any quantization types with zero dimensions and increment the embedding index accordingly.  
  
#### Challenge: Ensuring Correct Output Dimensions  
  
- **Issue**: When the `expand` flag is `True`, the actual dimensions of the embeddings increase due to expansion. We must ensure that the final embedding matches the requested `output_dim`.  
- **Solution**: Calculate the `expanded_dims_to_take` considering the expansion factor and adjust the slicing of embeddings accordingly.  
  
## Conclusion  
  
By customizing the Matryoshka Embedding Model to support multiple quantization methods across different embedding segments, we achieved greater flexibility and control over the embedding space. The hierarchical nesting property allows for efficient storage and retrieval, while the ability to specify dimensions and quantization types caters to various application requirements.  
  
Throughout the implementation, we addressed challenges related to embedding segmentation, cumulative to non-cumulative conversion, and dimension selection. The mathematical foundations provided a solid basis for the quantization and expansion mechanisms, ensuring that the model performs as intended.  
  
---  







