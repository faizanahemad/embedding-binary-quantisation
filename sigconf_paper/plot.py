import matplotlib.pyplot as plt
import numpy as np

# Data for Modern BERT (MB)
mb_dims = np.array([96, 192, 256, 384, 768])  # Actual dimensions
mb_data = {
    '1-bit': np.array([0.2908, 0.3285, 0.3691, 0.4167, 0.4381]),
    '1.5-bit': np.array([0.3455, 0.3901, 0.4228, 0.4429, 0.4536]),
    'Hybrid': np.array([0.3850, 0.4245, 0.4465, 0.4509, 0.4680]),
    '2-bit': np.array([0.3919, 0.4327, 0.4513, 0.4593, 0.4687]),
    'FP32': np.array([0.4247, 0.4512, 0.4680, 0.4695, 0.4720])
}

# Data for MiniLM
minilm_dims = np.array([48, 96, 128, 192, 384])  # Actual dimensions
minilm_data = {
    '1-bit': np.array([0.2687, 0.3417, 0.3571, 0.3724, 0.3839]),
    '1.5-bit': np.array([0.2871, 0.3649, 0.3814, 0.3923, 0.4101]),
    'Hybrid': np.array([0.2919, 0.3695, 0.3865, 0.4017, 0.4160]),
    '2-bit': np.array([0.2897, 0.3712, 0.3917, 0.4109, 0.4185]),
    'FP32': np.array([0.3014, 0.3792, 0.3963, 0.4219, 0.4286])
}

# Set figure size and style
print(plt.style.available)
plt.style.use('seaborn-v0_8-paper')
plt.figure(figsize=(12, 8))

# Line styles and markers
line_styles = {
    '1-bit': ('--', 'o'),
    '1.5-bit': ('-.', 's'),
    'Hybrid': (':', 'x'),
    '2-bit': ('-', '^'),
    'FP32': ('-', None)  # Solid line for FP32
}

color_map_mb = {
    '1-bit': '#90EE90',      # Light green
    '1.5-bit': '#32CD32',    # Lime green
    'Hybrid': '#228B22',     # Forest green
    '2-bit': '#006400',      # Dark green
    'FP32': '#005500'        # Dark green
}

color_map_minilm = {
    '1-bit': '#FFB6C1',      # Light pink
    '1.5-bit': '#DC143C',    # Crimson
    'Hybrid': '#B22222',     # Fire brick
    '2-bit': '#8B0000',      # Dark red
    'FP32': '#800000'        # Dark red but lighter than before
}

# Plot Modern BERT
max_dim_mb = max(mb_dims)
mb_handles = []  # Store handles for MB legend
for quant_type, values in mb_data.items():
    x = mb_dims / max_dim_mb
    y = values
    linestyle, marker = line_styles[quant_type]
    
    # Special handling for FP32
    if quant_type == 'FP32':
        linewidth = 3.0
        alpha = 1.0
    else:
        linewidth = 2.0
        alpha = 0.7
    
    line, = plt.plot(x, y,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=8,
                    color=color_map_mb[quant_type],
                    alpha=alpha,
                    label=f'MB-{quant_type}',
                    linewidth=linewidth)
    mb_handles.append(line)


# Plot MiniLM
max_dim_minilm = max(minilm_dims)
minilm_handles = []  # Store handles for MiniLM legend
for quant_type, values in minilm_data.items():
    x = minilm_dims / max_dim_minilm
    y = values
    linestyle, marker = line_styles[quant_type]
    
    # Special handling for FP32
    if quant_type == 'FP32':
        linewidth = 3.0
        alpha = 1.0
    else:
        linewidth = 2.0
        alpha = 0.7
    
    line, = plt.plot(x, y,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=8,
                    color=color_map_minilm[quant_type],
                    alpha=alpha,
                    label=f'MiniLM-{quant_type}',
                    linewidth=linewidth)
    minilm_handles.append(line)

# Customize the plot
plt.xlabel('Relative Dimension (D/D_max)', fontsize=12, fontweight='bold')
plt.ylabel('nDCG@10', fontsize=12, fontweight='bold')
plt.title('Impact of Embedding Dimensions on Performance', fontsize=14, fontweight='bold', pad=15)

# Set y-axis limits
plt.ylim(0.25, 0.5)

# Set x-axis ticks
plt.xticks([0.125, 0.25, 0.33, 0.5, 1.0],
           ['D/8', 'D/4', 'D/3', 'D/2', 'D'])

# Create two separate legends
# First legend for Modern BERT
leg1 = plt.legend(handles=mb_handles, title='Modern BERT',
                 bbox_to_anchor=(1.05, 1), loc='upper left',
                 borderaxespad=0., title_fontsize=12, fontsize=10)
# Add the second legend for MiniLM
plt.legend(handles=minilm_handles, title='MiniLM',
          bbox_to_anchor=(1.05, 0.5), loc='center left',
          borderaxespad=0., title_fontsize=12, fontsize=10)
# Add the first legend back
plt.gca().add_artist(leg1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()