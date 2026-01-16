import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
import os

class FITSCropTool:
    def __init__(self, fits_file):
        self.fits_file = fits_file
        self.original_data = None
        self.cropped_data = None
        self.crop_coords = None
        self.crop_history = []  # 记录每次crop的信息
        
        # Load FITS data
        with fits.open(fits_file) as hdul:
            self.original_data = hdul[0].data
            self.header = hdul[0].header.copy()
        
        print(f"Loaded FITS file: {fits_file}")
        print(f"Image shape: {self.original_data.shape}")
        
        # Apply ZScale normalization
        self.zscale = ZScaleInterval()
        self.vmin, self.vmax = self.zscale.get_limits(self.original_data)
        
        # 初始化：计算原始文件的histogram和拟合统计
        print("Analyzing original image...")
        self.current_fit_stats = self.calculate_fit_stats(self.original_data)
        self.crop_history.append({
            'coords': (0, self.original_data.shape[0], self.original_data.shape[1], 0),
            'label': 'Original',
            'fit_score': self.current_fit_stats['fit_score'],
            'chi_square': self.current_fit_stats['reduced_chi_square'],
            'r_squared': self.current_fit_stats['r_squared']
        })
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(20, 7))
        
        # Left: Input boxes and controls (0-1.5 columns)
        self.ax_inputs = plt.subplot2grid((1, 11), (0, 0), colspan=2)
        self.ax_inputs.axis('off')
        
        # Middle-Left: Record area (1.5-2.5 columns) - 窄窄的区域
        self.ax_record = plt.subplot2grid((1, 11), (0, 2), colspan=1)
        self.ax_record.axis('off')
        self.ax_record.set_xlim(0, 1)
        self.ax_record.set_ylim(0, 1)
        
        # Middle: Image display (3-6 columns)
        self.ax_image = plt.subplot2grid((1, 11), (0, 3), colspan=4)
        
        # Right: Histogram (7-11 columns)
        self.ax_hist = plt.subplot2grid((1, 11), (0, 7), colspan=4)
        
        # Display image
        self.im = self.ax_image.imshow(self.original_data, 
                                        cmap='gray', 
                                        origin='lower',
                                        vmin=self.vmin, 
                                        vmax=self.vmax,
                                        interpolation='nearest')
        self.ax_image.set_xlabel('X (pixels)')
        self.ax_image.set_ylabel('Y (pixels)')
        self.ax_image.set_title('FITS Image (ZScale + Linear)')
        
        # Colorbar
        plt.colorbar(self.im, ax=self.ax_image, label='Pixel Value')
        
        # Initialize record display
        self.record_text = self.ax_record.text(0.05, 0.95, 'Crop Record:', 
                                               transform=self.ax_record.transAxes,
                                               fontsize=9, fontweight='bold',
                                               verticalalignment='top', family='monospace')
        
        # Rectangle for crop region
        self.rect = None
        
        # Create input boxes
        self.create_input_widgets()
        
        # Initial histogram
        self.update_histogram(self.original_data)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_fit_stats(self, data):
        """计算histogram的拟合统计信息"""
        # 数据预处理
        pixel_values = data.flatten()
        pixel_values = pixel_values[np.isfinite(pixel_values)]
        
        # 数据裁剪：3000-7000范围
        threshold = 7000
        histogram_values = pixel_values[(pixel_values >= 3000) & (pixel_values < threshold)]
        
        if len(histogram_values) == 0:
            return {
                'r_squared': 0,
                'reduced_chi_square': np.inf,
                'fit_score': 0,
                'mean': 0,
                'std': 0
            }
        
        # 生成histogram
        n, bins = np.histogram(histogram_values, bins=2500, density=True)
        
        # Gaussian拟合
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_guess = np.mean(histogram_values)
        std_guess = np.std(histogram_values)
        initial_guess = [np.max(n), mean_guess, std_guess]
        
        try:
            popt, pcov = curve_fit(gaussian, bin_centers, n, p0=initial_guess)
            fit_amplitude, fit_mean, fit_std = popt
            
            # 计算拟合优度
            fitted_values = gaussian(bin_centers, fit_amplitude, fit_mean, fit_std)
            
            # R²
            ss_res = np.sum((n - fitted_values) ** 2)
            ss_tot = np.sum((n - np.mean(n)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Reduced χ²
            valid_mask = fitted_values > 1e-10
            chi_square = np.sum((n[valid_mask] - fitted_values[valid_mask]) ** 2 / fitted_values[valid_mask])
            n_params = 3
            dof = np.sum(valid_mask) - n_params
            reduced_chi_square = chi_square / dof if dof > 0 else np.inf
            
            # Fit Score
            fit_score = 100 * np.exp(-abs(reduced_chi_square - 1.0)) if reduced_chi_square < 100 else 0
            
            return {
                'r_squared': r_squared,
                'reduced_chi_square': reduced_chi_square,
                'fit_score': fit_score,
                'mean': fit_mean,
                'std': fit_std
            }
        except:
            return {
                'r_squared': 0,
                'reduced_chi_square': np.inf,
                'fit_score': 0,
                'mean': 0,
                'std': 0
            }
    
    def create_input_widgets(self):
        # Text instructions
        self.ax_inputs.text(0.1, 0.95, 'Crop Parameters:', 
                           fontsize=12, fontweight='bold',
                           transform=self.ax_inputs.transAxes)
        
        # Create text boxes for coordinates (positioned on left side)
        ax_x1 = plt.axes([0.02, 0.80, 0.08, 0.04])
        ax_y1 = plt.axes([0.02, 0.74, 0.08, 0.04])
        ax_x2 = plt.axes([0.02, 0.66, 0.08, 0.04])
        ax_y2 = plt.axes([0.02, 0.60, 0.08, 0.04])
        
        self.ax_inputs.text(0.1, 0.82, 'Top-Left X:', 
                           transform=self.ax_inputs.transAxes, fontsize=10)
        self.ax_inputs.text(0.1, 0.74, 'Top-Left Y:', 
                           transform=self.ax_inputs.transAxes, fontsize=10)
        self.ax_inputs.text(0.1, 0.66, 'Bottom-Right X:', 
                           transform=self.ax_inputs.transAxes, fontsize=10)
        self.ax_inputs.text(0.1, 0.58, 'Bottom-Right Y:', 
                           transform=self.ax_inputs.transAxes, fontsize=10)
        
        self.textbox_x1 = TextBox(ax_x1, '', initial='0')
        self.textbox_y1 = TextBox(ax_y1, '', initial=str(self.original_data.shape[0]))
        self.textbox_x2 = TextBox(ax_x2, '', initial=str(self.original_data.shape[1]))
        self.textbox_y2 = TextBox(ax_y2, '', initial='0')
        
        # Create buttons (positioned on left side)
        ax_confirm = plt.axes([0.02, 0.50, 0.08, 0.05])
        ax_save = plt.axes([0.02, 0.43, 0.08, 0.05])
        ax_reset = plt.axes([0.02, 0.36, 0.08, 0.05])
        
        self.btn_confirm = Button(ax_confirm, 'Confirm Crop')
        self.btn_save = Button(ax_save, 'Save Cropped')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        # Connect button callbacks
        self.btn_confirm.on_clicked(self.confirm_crop)
        self.btn_save.on_clicked(self.save_cropped)
        self.btn_reset.on_clicked(self.reset_view)
        
        # Status text
        self.status_text = self.ax_inputs.text(0.1, 0.25, '', 
                                               transform=self.ax_inputs.transAxes,
                                               fontsize=8, color='blue', wrap=True)
        
        # Info text
        info_text = (f"Image size:\n{self.original_data.shape[1]} x {self.original_data.shape[0]}\n"
                    f"ZScale range:\n[{self.vmin:.1f}, {self.vmax:.1f}]")
        self.ax_inputs.text(0.1, 0.10, info_text, 
                           transform=self.ax_inputs.transAxes,
                           fontsize=7, family='monospace')
    
    def confirm_crop(self, event):
        try:
            # Get coordinates from textboxes
            x1 = int(self.textbox_x1.text)
            y1 = int(self.textbox_y1.text)
            x2 = int(self.textbox_x2.text)
            y2 = int(self.textbox_y2.text)
            
            # Validate coordinates
            h, w = self.original_data.shape
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            
            # Ensure x1 < x2 and y2 < y1 (top-left to bottom-right)
            if x1 >= x2 or y2 >= y1:
                self.status_text.set_text('Error: Invalid coordinates!\nX1<X2, Y2<Y1')
                self.status_text.set_color('red')
                self.fig.canvas.draw()
                return
            
            # Store crop coordinates
            self.crop_coords = (x1, y1, x2, y2)
            
            # Crop the data (note: numpy array is [row, col] = [y, x])
            self.cropped_data = self.original_data[y2:y1, x1:x2]
            
            # Draw rectangle on image
            if self.rect:
                self.rect.remove()
            
            width = x2 - x1
            height = y1 - y2
            self.rect = Rectangle((x1, y2), width, height, 
                                 linewidth=2, edgecolor='red', 
                                 facecolor='none', linestyle='-')
            self.ax_image.add_patch(self.rect)
            
            # Update histogram with cropped data
            self.update_histogram(self.cropped_data)
            
            # 添加到历史记录
            fit_stats = getattr(self, 'current_fit_stats', {'fit_score': 0, 'reduced_chi_square': np.inf, 'r_squared': 0})
            self.crop_history.append({
                'coords': (x1, y1, x2, y2),
                'fit_score': fit_stats['fit_score'],
                'chi_square': fit_stats['reduced_chi_square'],
                'r_squared': fit_stats['r_squared']
            })
            
            # 更新record显示
            self.update_record_display()
            
            # Update status
            crop_size = self.cropped_data.shape
            self.status_text.set_text(f'Cropped: {crop_size[1]}x{crop_size[0]} pixels\n'
                                     f'Region: [{x1}:{x2}, {y2}:{y1}]')
            self.status_text.set_color('green')
            
            self.fig.canvas.draw()
            
        except Exception as e:
            self.status_text.set_text(f'Error: {str(e)}')
            self.status_text.set_color('red')
            self.fig.canvas.draw()
    
    def update_histogram(self, data):
        """
        更新histogram显示，包含Gaussian拟合
        
        完整流程：
        1. 数据预处理：展平数组，移除无效值
        2. 数据裁剪：去除极端高值（超过mean+5σ）以聚焦背景噪声
        3. 生成histogram：归一化频率分布
        4. Gaussian拟合：使用curve_fit优化拟合参数
        5. 计算拟合优度：R²评估拟合质量
        6. 绘制结果：显示histogram、拟合曲线和背景均值线
        """
        self.ax_hist.clear()
        
        # ========== 步骤1：数据预处理 ==========
        # 将2D图像数据展平为1D数组，便于统计分析
        pixel_values = data.flatten()
        # 移除NaN和无穷值，确保数据有效性
        pixel_values = pixel_values[np.isfinite(pixel_values)]
        
        # ========== 步骤2：数据裁剪 ==========
        # 设置阈值：去除>7000的超亮像素（亮星和blooming）
        # 这样可以更好地看到背景噪声的Gaussian分布
        threshold = 5800
        suppressed_values = pixel_values[pixel_values < threshold]
        
        # 进一步筛选：只保留3000-7000范围内的像素用于histogram显示
        histogram_values = suppressed_values[(suppressed_values >= 3000) & (suppressed_values < 5800)]
        
        # ========== 步骤3：生成histogram ==========
        # bins=2500: 将3000-7000范围分成2500个区间
        # density=True: 归一化，使histogram积分为1（概率密度）
        # 返回值：n是每个bin的频率，bins是bin边界，patches是绘图对象
        n, bins, patches = self.ax_hist.hist(histogram_values, bins=2550, 
                                             color='blue', alpha=0.7, 
                                             density=True, 
                                             label='Pixel distribution')
        
        # ========== 步骤4：Gaussian拟合 ==========
        # 定义Gaussian函数：amplitude * exp(-(x-mean)²/(2*σ²))
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
        
        # 计算每个bin的中心位置（用于拟合）
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 设置拟合初始猜测值（使用histogram范围内的数据）
        mean_guess = np.mean(histogram_values)  # 均值作为中心
        std_guess = np.std(histogram_values)     # 标准差作为宽度
        initial_guess = [np.max(n), mean_guess, std_guess]  # [振幅, 均值, 标准差]
        
        try:
            # ========== curve_fit进行非线性最小二乘拟合 ==========
            # popt: 最优参数 [amplitude, mean, stddev]
            # pcov: 参数协方差矩阵（用于计算误差）
            popt, pcov = curve_fit(gaussian, bin_centers, n, p0=initial_guess)
            fit_amplitude, fit_mean, fit_std = popt
            
            # 计算参数误差：协方差矩阵对角线的平方根
            perr = np.sqrt(np.diag(pcov))
            
            # ========== 步骤5：计算拟合优度 ==========
            # 在bin中心计算拟合值
            fitted_values = gaussian(bin_centers, fit_amplitude, fit_mean, fit_std)
            
            # 1. R²（决定系数）- 对峰值敏感
            ss_res = np.sum((n - fitted_values) ** 2)  # 残差平方和
            ss_tot = np.sum((n - np.mean(n)) ** 2)     # 总平方和
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 2. Reduced Chi-square (χ²ᵣ) - 更敏感于局部偏差
            # χ²ᵣ = Σ[(observed - expected)² / expected] / (n_bins - n_params)
            # 理想值接近1.0，>1表示拟合不佳，<1表示过度拟合
            # 避免除以零，只计算fitted_values > 1e-10的bins
            valid_mask = fitted_values > 1e-10
            chi_square = np.sum((n[valid_mask] - fitted_values[valid_mask]) ** 2 / fitted_values[valid_mask])
            n_params = 3  # 拟合参数数量: amplitude, mean, stddev
            dof = np.sum(valid_mask) - n_params  # 自由度
            reduced_chi_square = chi_square / dof if dof > 0 else np.inf
            
            # 计算拟合质量评分（0-100）
            # Reduced χ²越接近1越好，转换为0-100分数
            # score = 100 * exp(-|χ²ᵣ - 1|)
            fit_score = 100 * np.exp(-abs(reduced_chi_square - 1.0)) if reduced_chi_square < 100 else 0
            
            # 存储当前拟合统计信息
            self.current_fit_stats = {
                'r_squared': r_squared,
                'reduced_chi_square': reduced_chi_square,
                'fit_score': fit_score,
                'mean': fit_mean,
                'std': fit_std
            }
            
            # ========== 步骤6：绘制拟合曲线 ==========
            # 生成平滑的x值用于绘制拟合曲线
            x = np.linspace(bins[0], bins[-1], 1000)
            fitted_gaussian = gaussian(x, fit_amplitude, fit_mean, fit_std)
            self.ax_hist.plot(x, fitted_gaussian, 'r-', linewidth=2, 
                             label=f'Gaussian fit\nμ={fit_mean:.2f}±{perr[1]:.2f}\nσ={fit_std:.2f}±{perr[2]:.2f}\n'
                                   f'R²={r_squared:.4f}\nχ²ᵣ={reduced_chi_square:.3f}\nScore={fit_score:.1f}/100')
            
            # 绘制均值线（背景水平）
            self.ax_hist.axvline(fit_mean, color='green', linestyle='--', 
                                linewidth=2, alpha=0.7)
            
        except Exception as e:
            # 如果拟合失败，静默处理
            print(f"Gaussian fit failed: {e}")
            self.current_fit_stats = {
                'r_squared': 0,
                'reduced_chi_square': np.inf,
                'fit_score': 0,
                'mean': 0,
                'std': 0
            }
            pass
        
        self.ax_hist.set_xlabel('Pixel Value')
        self.ax_hist.set_ylabel('Normalized Frequency')
        self.ax_hist.set_title(f'Histogram (Range: 3000-7000, suppressed > {threshold})')
        self.ax_hist.set_xlim(3200, 3900)  # Set x-axis range to 3000-7000
        self.ax_hist.legend(fontsize=9)
        self.ax_hist.grid(True, alpha=0.3)
    
    def save_cropped(self, event):
        if self.cropped_data is None or self.crop_coords is None:
            self.status_text.set_text('Error: No crop defined!\nClick Confirm first.')
            self.status_text.set_color('red')
            self.fig.canvas.draw()
            return
        
        try:
            # Create output filename
            x1, y1, x2, y2 = self.crop_coords
            base_path = os.path.dirname(self.fits_file)
            base_name = os.path.basename(self.fits_file)
            name_part = os.path.splitext(base_name)[0]
            
            output_file = os.path.join(base_path, 
                                      f"{name_part}_crop_x{x1}-{x2}_y{y2}-{y1}.fits")
            
            # Update header with crop information
            self.header['COMMENT'] = f'Cropped from {base_name}'
            self.header['CROPX1'] = (x1, 'Crop region X start')
            self.header['CROPX2'] = (x2, 'Crop region X end')
            self.header['CROPY1'] = (y2, 'Crop region Y start')
            self.header['CROPY2'] = (y1, 'Crop region Y end')
            self.header['CROPPED'] = (True, 'Image has been cropped')
            
            # Save cropped FITS file
            hdu = fits.PrimaryHDU(data=self.cropped_data, header=self.header)
            hdu.writeto(output_file, overwrite=True)
            
            self.status_text.set_text(f'Saved successfully!\n{os.path.basename(output_file)}')
            self.status_text.set_color('green')
            print(f"\nCropped FITS saved: {output_file}")
            print(f"Crop region: X[{x1}:{x2}], Y[{y2}:{y1}]")
            print(f"Cropped size: {self.cropped_data.shape}")
            
            self.fig.canvas.draw()
            
        except Exception as e:
            self.status_text.set_text(f'Save error: {str(e)}')
            self.status_text.set_color('red')
            self.fig.canvas.draw()
    
    def update_record_display(self):
        """更新record区域显示最近3次crop记录（4行格式）"""
        self.ax_record.clear()
        self.ax_record.axis('off')
        self.ax_record.set_xlim(0, 1)
        self.ax_record.set_ylim(0, 1)
        
        # 显示标题
        self.ax_record.text(0.05, 0.98, 'Crop Records:', 
                           transform=self.ax_record.transAxes,
                           fontsize=8, fontweight='bold',
                           verticalalignment='top', family='monospace')
        
        # 显示最近3次记录
        recent_crops = self.crop_history[-3:] if len(self.crop_history) > 0 else []
        y_pos = 0.88
        
        for i, record in enumerate(reversed(recent_crops), 1):
            x1, y1, x2, y2 = record['coords']
            score = record['fit_score']
            chi2 = record['chi_square']
            r2 = record['r_squared']
            label = record.get('label', '')
            
            # 根据score设置颜色
            if score > 80:
                color = 'green'
            elif score > 50:
                color = 'orange'
            else:
                color = 'red'
            
            # 4行显示
            if label == 'Original':
                text = (f"Original Image\n"
                       f"Size:{x2}x{y1}\n"
                       f"R²={r2:.4f}\n"
                       f"χ²ᵣ={chi2:.3f}")
            else:
                text = (f"#{len(self.crop_history) - len(recent_crops) + i}:({x1},{y1})→({x2},{y2})\n"
                       f"R²={r2:.4f}\n"
                       f"χ²ᵣ={chi2:.3f}\n"
                       f"Score={score:.0f}/100")
            
            self.ax_record.text(0.05, y_pos, text,
                               transform=self.ax_record.transAxes,
                               fontsize=6, family='monospace',
                               verticalalignment='top',
                               color=color)
            y_pos -= 0.28
    
    def reset_view(self, event):
        # Remove rectangle
        if self.rect:
            self.rect.remove()
            self.rect = None
        
        # Reset cropped data
        self.cropped_data = None
        self.crop_coords = None
        
        # Reset histogram to original data
        self.update_histogram(self.original_data)
        
        # Reset textboxes
        self.textbox_x1.set_val('0')
        self.textbox_y1.set_val(str(self.original_data.shape[0]))
        self.textbox_x2.set_val(str(self.original_data.shape[1]))
        self.textbox_y2.set_val('0')
        
        # Clear status
        self.status_text.set_text('Reset to original view')
        self.status_text.set_color('blue')
        
        self.fig.canvas.draw()


def main():
    # Get script directory and construct path to FITS file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fits_file = os.path.join(project_root, 'Fits_Data', 'mosaic.fits')
    
    if not os.path.exists(fits_file):
        print(f"Error: FITS file not found at {fits_file}")
        return
    
    # Create and run the crop tool
    tool = FITSCropTool(fits_file)


if __name__ == '__main__':
    main()
