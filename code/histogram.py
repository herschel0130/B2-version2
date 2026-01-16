import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm
from scipy.optimize import curve_fit
import os

# Read the FITS file
# Get the script's directory and construct absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
fits_file = os.path.join(project_root, 'Fits_Data', 'mosaic.fits')

print(f"Looking for FITS file at: {fits_file}")
print(f"File exists: {os.path.exists(fits_file)}\n")

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    print(f"Image shape: {data.shape}")
    print(f"Data type: {data.dtype}")

# Flatten the 2D image data to 1D array
pixel_values = data.flatten()

# Remove NaN or infinite values if present
pixel_values = pixel_values[np.isfinite(pixel_values)]

# Calculate basic statistics
mean_val = np.mean(pixel_values)
median_val = np.median(pixel_values)
std_val = np.std(pixel_values)
min_val = np.min(pixel_values)
max_val = np.max(pixel_values)

print(f"\nBasic Statistics:")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Std Dev: {std_val:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Full histogram
ax1.hist(pixel_values, bins=3000, color='blue', alpha=0.7)
ax1.set_xlabel('Pixel Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Full Histogram of Pixel Values')
ax1.set_yscale('log')  # Log scale to see the tail better
ax1.grid(True, alpha=0.3)

# Plot 2: Histogram with suppressed high values to see Gaussian better
# Suppress values beyond 7000 (bright stars and blooming)
# Focus on 3000-7000 range for histogram display
threshold = 5500
suppressed_values = pixel_values[(pixel_values >= 3000) & (pixel_values < threshold)]

print(f"\nHistogram range: 3000 to {threshold}")
print(f"Percentage of pixels shown: {100 * len(suppressed_values) / len(pixel_values):.2f}%")

# Calculate statistics for the suppressed data
mean_suppressed = np.mean(suppressed_values)
std_suppressed = np.std(suppressed_values)

# Create histogram (归一化频率分布) - 2500 bins for 3000-7000 range
n, bins, patches = ax2.hist(suppressed_values, bins=2500, color='blue', 
                             alpha=0.7, edgecolor='black', density=True, 
                             label='Pixel distribution')

# ========== Gaussian拟合流程 ==========
# 定义Gaussian函数
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# 计算bin中心位置（用于拟合）
bin_centers = (bins[:-1] + bins[1:]) / 2

# 设置初始猜测值：[振幅, 均值, 标准差]
initial_guess = [np.max(n), mean_suppressed, std_suppressed]

try:
    # 使用curve_fit进行非线性最小二乘拟合
    popt, pcov = curve_fit(gaussian, bin_centers, n, p0=initial_guess)
    fit_amplitude, fit_mean, fit_std = popt
    
    # 计算参数误差（协方差矩阵对角线的平方根）
    perr = np.sqrt(np.diag(pcov))
    
    # ========== 计算拟合优度 ==========
    fitted_values = gaussian(bin_centers, fit_amplitude, fit_mean, fit_std)
    
    # 1. R²（决定系数）- 对峰值敏感
    ss_res = np.sum((n - fitted_values) ** 2)  # 残差平方和
    ss_tot = np.sum((n - np.mean(n)) ** 2)     # 总平方和
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  # R²决定系数
    
    # 2. Reduced Chi-square (χ²ᵣ) - 更敏感于局部偏差
    # χ²ᵣ = Σ[(observed - expected)² / expected] / (n_bins - n_params)
    # 理想值接近1.0，>1表示拟合不佳，<1表示过度拟合
    valid_mask = fitted_values > 1e-10
    chi_square = np.sum((n[valid_mask] - fitted_values[valid_mask]) ** 2 / fitted_values[valid_mask])
    n_params = 3  # 拟合参数数量: amplitude, mean, stddev
    dof = np.sum(valid_mask) - n_params  # 自由度
    reduced_chi_square = chi_square / dof if dof > 0 else np.inf
    
    # 计算拟合质量评分（0-100）
    fit_score = 100 * np.exp(-abs(reduced_chi_square - 1.0)) if reduced_chi_square < 100 else 0
    
    # 绘制拟合曲线
    x = np.linspace(bins[0], bins[-1], 2200)
    fitted_gaussian = gaussian(x, fit_amplitude, fit_mean, fit_std)
    ax2.plot(x, fitted_gaussian, 'r-', linewidth=2, 
             label=f'Gaussian fit\n(μ={fit_mean:.2f}±{perr[1]:.2f}, σ={fit_std:.2f}±{perr[2]:.2f})\n'
                   f'R²={r_squared:.4f}\nχ²ᵣ={reduced_chi_square:.3f}\nScore={fit_score:.1f}/100')
    
    ax2.axvline(fit_mean, color='green', linestyle='--', linewidth=2, 
                label=f'Fitted mean (background): {fit_mean:.2f}')
    
    print(f"\nGaussian Fit Results:")
    print(f"Fitted Mean: {fit_mean:.2f} ± {perr[1]:.2f}")
    print(f"Fitted Std Dev: {fit_std:.2f} ± {perr[2]:.2f}")
    print(f"Fitted Amplitude: {fit_amplitude:.6f} ± {perr[0]:.6f}")
    print(f"R² (goodness of fit): {r_squared:.4f}")
    print(f"Reduced χ²: {reduced_chi_square:.3f}")
    print(f"Fit Score: {fit_score:.1f}/100")
    
except Exception as e:
    print(f"\nGaussian fit failed: {e}")
    print("Using simple statistics instead")
    # Fallback to simple Gaussian using statistics
    x = np.linspace(bins[0], bins[-1], 2000)
    gaussian_simple = norm.pdf(x, mean_suppressed, std_suppressed)
    ax2.plot(x, gaussian_simple, 'r-', linewidth=2, 
             label=f'Gaussian (stats)\n(μ={mean_suppressed:.2f}, σ={std_suppressed:.2f})')
    
    ax2.axvline(mean_suppressed, color='green', linestyle='--', linewidth=2, 
                label=f'Mean (background): {mean_suppressed:.2f}')

ax2.set_xlabel('Pixel Value')
ax2.set_ylabel('Normalized Frequency')
ax2.set_title(f'Histogram (Range: 3000-7000, suppressed > {threshold})\nShowing Gaussian Background + Source Tail')
ax2.set_xlim(3000, 7000)  # Set x-axis range to 3000-7000
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(script_dir, 'histogram_analysis.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nHistogram saved as '{output_file}'")
plt.show()

# Additional analysis: Identify pixels significantly above background
# Pixels above mean + 3*sigma are likely from astronomical sources
detection_threshold = mean_val + 3 * std_val
source_pixels = pixel_values[pixel_values > detection_threshold]
print(f"\nSource Detection:")
print(f"Detection threshold (mean + 3σ): {detection_threshold:.2f}")
print(f"Number of pixels above threshold: {len(source_pixels)}")
print(f"Percentage of image: {100 * len(source_pixels) / len(pixel_values):.3f}%")
