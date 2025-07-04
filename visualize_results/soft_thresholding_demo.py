import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def soft_threshold(x, lambda_val):
    """
    Soft-thresholding operator:
    S_λ(x) = sign(x) * max{|x| - λ, 0}
    """
    return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

# Create discrete frequency domain data (like real FFT output)
n_freq = 36  # 36 discrete frequency components
freq_indices = np.arange(n_freq)

# Create high-resolution indices for smooth interpolation
n_smooth = 200
freq_indices_smooth = np.linspace(0, n_freq-1, n_smooth)

# Create a more exponentially decaying frequency signal with enhanced wiggliness
# Stronger exponential decay for more realistic atmospheric spectrum
base_decay = 3.0 * np.exp(-freq_indices / 5.0)  # Faster decay

# Add much more prominent oscillations (wiggly behavior)
# These can produce negative values representing phase information or atmospheric patterns
wiggles = 0.6 * np.sin(2 * np.pi * freq_indices / 4.5) + 0.4 * np.cos(2 * np.pi * freq_indices / 3.2)
wiggles += 0.3 * np.sin(2 * np.pi * freq_indices / 2.8 + np.pi/4)  # Additional phase component
wiggles += 0.25 * np.cos(2 * np.pi * freq_indices / 6.1 + np.pi/6)  # More complexity

# Add some random noise 
np.random.seed(42)  # For reproducibility
noise = 0.15 * np.random.randn(n_freq)

# Add a more prominent negative component (also exponentially decaying)
negative_component = -0.4 * np.exp(-freq_indices / 7.0) * np.sin(freq_indices / 1.8 + np.pi/3)

# Combine to create the original frequency domain signal (discrete components only!)
original_signal = base_decay + wiggles + noise + negative_component

# Create smooth interpolated version for plotting
f_interp = interp1d(freq_indices, original_signal, kind='cubic', bounds_error=False, fill_value='extrapolate')
original_signal_smooth = f_interp(freq_indices_smooth)

# Apply soft-thresholding with different lambda values
lambda_values = [0.2, 0.4, 0.6, 0.8]
thresholded_signals = {}

thresholded_signals_smooth = {}

for lam in lambda_values:
    # Apply thresholding to discrete frequency components
    thresholded_signals[lam] = soft_threshold(original_signal, lam)
    
    # Create smooth interpolated version of thresholded signal
    f_thresh = interp1d(freq_indices, thresholded_signals[lam], kind='cubic', bounds_error=False, fill_value='extrapolate')
    thresholded_signals_smooth[lam] = f_thresh(freq_indices_smooth)

# Create 2x2 grid for the 4 lambda threshold plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Soft-Thresholding in Frequency Domain (Wiggly Curves, 36 Discrete Components)', fontsize=16, fontweight='bold')

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Plot for each lambda value in 2x2 grid
for i, lam in enumerate(lambda_values):
    ax = axes_flat[i]
    
    # Plot smooth interpolated curves (connecting the 36 discrete frequency components)
    ax.plot(freq_indices_smooth, original_signal_smooth, 'blue', linewidth=2.5, alpha=0.8, 
            label='Original frequency')
    ax.plot(freq_indices_smooth, thresholded_signals_smooth[lam], 'red', linewidth=2.5, 
            label=f'Sparsified (λ={lam})')
    
    # Add threshold lines
    ax.axhline(y=lam, color='red', linestyle='--', alpha=0.5, label=f'+λ = {lam}')
    ax.axhline(y=-lam, color='red', linestyle='--', alpha=0.5, label=f'-λ = {lam}')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Count how many components were zeroed
    zeros = np.sum(np.abs(thresholded_signals[lam]) == 0)

    
    #ax.set_title(f'λ = {lam}: {zeros}/{len(original_signal)} components → 0\n(Wiggly exponential decay, {len(original_signal)} discrete frequencies)')
    ax.set_xlabel('Frequency Index', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)  # Increase tick label size

plt.tight_layout()

# Save the plot
plt.savefig('/mydata/deepcloud/yves/results_final/soft_thresholding_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics for discrete frequency components
print("=== Soft-Thresholding Analysis ===")
print(f"Original signal: {len(original_signal)} frequency components")
print(f"Original signal range: [{np.min(original_signal):.3f}, {np.max(original_signal):.3f}]")
print(f"Original signal energy: {np.sum(original_signal**2):.3f}")
print()

for lam in lambda_values:
    # Use discrete frequency components for statistics
    thresholded = thresholded_signals[lam]
    zeros = np.sum(np.abs(thresholded) == 0)
    energy = np.sum(thresholded**2)
    energy_ratio = energy / np.sum(original_signal**2) * 100
    
    print(f"λ = {lam}:")
    print(f"  Components set to zero: {zeros}/{len(thresholded)} ({zeros/len(thresholded)*100:.1f}%)")
    print(f"  Remaining energy: {energy_ratio:.1f}% of original")
    print(f"  Range: [{np.min(thresholded):.3f}, {np.max(thresholded):.3f}]")
    print()

print("The soft-thresholding operator:")
print("• Removes components with |amplitude| ≤ λ")
print("• Shrinks larger components by λ")
print("• Preserves the sign of the original components")
print("• Promotes sparsity in the frequency domain")

# Save analysis results to text file
with open('/mydata/deepcloud/yves/results_final/soft_thresholding_analysis.txt', 'w') as f:
    f.write("=== Soft-Thresholding Analysis ===\n")
    f.write(f"Original signal: {len(original_signal)} frequency components\n")
    f.write(f"Original signal range: [{np.min(original_signal):.3f}, {np.max(original_signal):.3f}]\n")
    f.write(f"Original signal energy: {np.sum(original_signal**2):.3f}\n\n")
    
    for lam in lambda_values:
        thresholded = thresholded_signals[lam]
        zeros = np.sum(np.abs(thresholded) == 0)
        energy = np.sum(thresholded**2)
        energy_ratio = energy / np.sum(original_signal**2) * 100
        
        f.write(f"λ = {lam}:\n")
        f.write(f"  Components set to zero: {zeros}/{len(thresholded)} ({zeros/len(thresholded)*100:.1f}%)\n")
        f.write(f"  Remaining energy: {energy_ratio:.1f}% of original\n")
        f.write(f"  Range: [{np.min(thresholded):.3f}, {np.max(thresholded):.3f}]\n\n")
    
    f.write("The soft-thresholding operator:\n")
    f.write("• Removes components with |amplitude| ≤ λ\n")
    f.write("• Shrinks larger components by λ\n")
    f.write("• Preserves the sign of the original components\n")
    f.write("• Promotes sparsity in the frequency domain\n")

print(f"\nFiles saved to /mydata/deepcloud/yves/results_final/:")
print("- soft_thresholding_demo.png (wiggly interpolated curves)")
print("- soft_thresholding_analysis.txt (analysis of 36 discrete frequency components)")
print()
print("Key features:")
print("• 36 discrete frequency components with wiggly exponential decay")
print("• Smooth interpolated curves connecting only the discrete frequencies")
print("• Enhanced wiggliness with multiple oscillatory components")
print("• No artificial peaks between discrete frequency indices")
print("• 2x2 grid showing each λ value with original vs sparsified")
print("• λ values: [0.2, 0.4, 0.6, 0.8] for progressive sparsification")
print("• Clear demonstration that |x| ≤ λ → x = 0 (complete zeroing)") 