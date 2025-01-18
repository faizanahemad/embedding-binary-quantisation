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
plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

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

# Add baseline data for Modern BERT
mb_baseline_data = {
    '1-bit': np.array([0.2500, 0.2900, 0.3300, 0.3900, 0.4350]),  # Thresholds Only row
    '2-bit': np.array([0.3500, 0.3850, 0.4200, 0.4370, 0.4640])   # Thresholds Only row
}

# Add baseline data for MiniLM
minilm_baseline_data = {
    '1-bit': np.array([0.1180, 0.2050, 0.2566, 0.2947, 0.3055]),  # Thresholds Only row
    '2-bit': np.array([0.1500, 0.2428, 0.2879, 0.3324, 0.3431])   # Thresholds Only row
}

# Update color maps to include baseline colors
color_map_mb.update({
    '1-bit-baseline': '#B19CD9',  # Light purplish
    '2-bit-baseline': '#9370DB'   # Medium purplish
})

color_map_minilm.update({
    '1-bit-baseline': '#FFD700',  # Darker yellow (Gold)
    '2-bit-baseline': '#FFB347'   # Darker orange/yellow
})

# Plot Modern BERT on first subplot
max_dim_mb = max(mb_dims)
mb_handles = []  # Store handles for MB legend

# Plot baselines first
for quant_type in ['1-bit']:
    x = mb_dims / max_dim_mb
    y = mb_baseline_data[quant_type]
    line, = ax1.plot(x, y,
                    linestyle='dashdot',
                    marker=None,
                    color=color_map_mb[f'{quant_type}-baseline'],
                    alpha=0.6,
                    label=f'MB-{quant_type}-baseline',
                    linewidth=1.5)
    mb_handles.append(line)
    
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
    
    line, = ax1.plot(x, y,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=8,
                    color=color_map_mb[quant_type],
                    alpha=alpha,
                    label=f'MB-{quant_type}',
                    linewidth=linewidth)
    mb_handles.append(line)

max_dim_minilm = max(minilm_dims)
minilm_handles = []  # Store handles for MiniLM legend

# Plot baselines first
for quant_type in ['1-bit', ]:
    x = minilm_dims / max_dim_minilm
    y = minilm_baseline_data[quant_type]
    line, = ax2.plot(x, y,
                    linestyle='dashdot',
                    marker=None,
                    color=color_map_minilm[f'{quant_type}-baseline'],
                    alpha=0.6,
                    label=f'MiniLM-{quant_type}-baseline',
                    linewidth=1.5)
    minilm_handles.append(line)

# Plot MiniLM on second subplot

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
    
    line, = ax2.plot(x, y,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=8,
                    color=color_map_minilm[quant_type],
                    alpha=alpha,
                    label=f'MiniLM-{quant_type}',
                    linewidth=linewidth)
    minilm_handles.append(line)

# Customize the plots
for ax in [ax1, ax2]:
    ax.set_xlabel('Relative Dimension (D/D_max)', fontsize=12)
    ax.set_ylabel('nDCG@10', fontsize=12, fontweight='bold')
    if ax == ax1:
        ax.set_ylim(0.2, 0.5)
    else:
        ax.set_ylim(0.1, 0.45)
    ax.set_xticks([0.125, 0.25, 0.33, 0.5, 1.0])
    ax.set_xticklabels(['D/8', 'D/4', 'D/3', 'D/2', 'D'])
    ax.grid(True, linestyle='--', alpha=0.7)

# Set specific titles for each subplot
ax1.set_title('Modern BERT: Impact of Embedding Dimensions on Performance', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_title('MiniLM: Impact of Embedding Dimensions on Performance', 
              fontsize=14, fontweight='bold', pad=15)

# Create legends for each subplot
ax1.legend(handles=mb_handles, title='Modern BERT Quantization',
          bbox_to_anchor=(1.05, 1), loc='upper left',
          borderaxespad=0., title_fontsize=12, fontsize=10)

ax2.legend(handles=minilm_handles, title='MiniLM Quantization',
          bbox_to_anchor=(1.05, 1), loc='upper left',
          borderaxespad=0., title_fontsize=12, fontsize=10)

# Add a main title for the entire figure
fig.suptitle('', # Comparison of Embedding Dimension Impact on Model Performance
            fontsize=16, fontweight='bold', y=0.98)

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.subplots_adjust(top=0.93)

# Show plot
plt.show()