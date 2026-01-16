import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
import os

class FITSCropToolTerminal:
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
        print("\nAnalyzing original image...")
        self.current_fit_stats = self.calculate_fit_stats(self.original_data)
        self.crop_history.append({
            'coords': (0, self.original_data.shape[0], self.original_data.shape[1], 0),
            'label': 'Original',
            'fit_score': self.current_fit_stats['fit_score'],
            'chi_square': self.current_fit_stats['reduced_chi_square'],
            'r_squared': self.current_fit_stats['r_squared']
        })
        
        # Create figure
        self.setup_figure()
        
    def setup_figure(self):
        """设置图形界面（无操作框）"""
        self.fig = plt.figure(figsize=(18, 7))
        
        # Left: Record area
        self.ax_record = plt.subplot2grid((1, 10), (0, 0), colspan=2)
        self.ax_record.axis('off')
        
        # Middle: Image display
        self.ax_image = plt.subplot2grid((1, 10), (0, 2), colspan=4)
        
        # Right: Histogram
        self.ax_hist = plt.subplot2grid((1, 10), (0, 6), colspan=4)
        
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
        
        # Rectangle for crop region
        self.rect = None
        
        # Initial display
        self.update_histogram(self.original_data)
        self.update_record_display()
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def calculate_fit_stats(self, data):
        """计算histogram的拟合统计信息"""
        # 数据预处理
        pixel_values = data.flatten()
        pixel_values = pixel_values[np.isfinite(pixel_values)]
        
        # 数据裁剪：3000-5800范围
        threshold = 5800
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
    
    def update_histogram(self, data):
        """更新histogram显示"""
        self.ax_hist.clear()
        
        # 数据预处理
        pixel_values = data.flatten()
        pixel_values = pixel_values[np.isfinite(pixel_values)]
        
        # 数据裁剪
        threshold = 5800
        histogram_values = pixel_values[(pixel_values >= 3000) & (pixel_values < threshold)]
        
        # 生成histogram
        n, bins, patches = self.ax_hist.hist(histogram_values, bins=2500, 
                                             color='blue', alpha=0.7, 
                                             density=True, 
                                             label='Pixel distribution')
        
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
            perr = np.sqrt(np.diag(pcov))
            
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
            
            # 存储统计信息
            self.current_fit_stats = {
                'r_squared': r_squared,
                'reduced_chi_square': reduced_chi_square,
                'fit_score': fit_score,
                'mean': fit_mean,
                'std': fit_std
            }
            
            # 绘制拟合曲线
            x = np.linspace(bins[0], bins[-1], 2500)
            fitted_gaussian = gaussian(x, fit_amplitude, fit_mean, fit_std)
            self.ax_hist.plot(x, fitted_gaussian, 'r-', linewidth=2, 
                             label=f'Gaussian fit\nμ={fit_mean:.2f}±{perr[1]:.2f}\nσ={fit_std:.2f}±{perr[2]:.2f}\n'
                                   f'R²={r_squared:.4f}\nχ²ᵣ={reduced_chi_square:.3f}\nScore={fit_score:.1f}/100')
            
            self.ax_hist.axvline(fit_mean, color='green', linestyle='--', linewidth=2, alpha=0.7)
            

        except Exception as e:
            print(f"Gaussian fit failed: {e}")
            self.current_fit_stats = {
                'r_squared': 0,
                'reduced_chi_square': np.inf,
                'fit_score': 0,
                'mean': 0,
                'std': 0
            }
        
        self.ax_hist.set_xlabel('Pixel Value')
        self.ax_hist.set_ylabel('Normalized Frequency')
        self.ax_hist.set_title(f'Histogram (Range: 3000-7000, suppressed > {threshold})')
        self.ax_hist.set_xlim(3000, 7000)
        self.ax_hist.legend(fontsize=9)
        self.ax_hist.grid(True, alpha=0.3)
    
    def update_record_display(self):
        """更新record区域显示所有crop记录（4行格式，可滚动）"""
        self.ax_record.clear()
        self.ax_record.axis('off')
        self.ax_record.set_xlim(0, 1)
        self.ax_record.set_ylim(0, 1)
        
        # 显示标题
        title = f'Crop Records ({len(self.crop_history)} total):'
        self.ax_record.text(0.05, 0.98, title, 
                           transform=self.ax_record.transAxes,
                           fontsize=8, fontweight='bold',
                           verticalalignment='top', family='monospace')
        
        # 计算可显示的记录数（每个记录约0.26的高度）
        available_height = 0.85  # 去除标题后的可用高度
        record_height = 0.26
        max_visible = int(available_height / record_height)
        
        # 显示最近的记录（从最新到最旧）
        visible_crops = self.crop_history[-max_visible:] if len(self.crop_history) > max_visible else self.crop_history
        y_pos = 0.88
        
        for i, record in enumerate(reversed(visible_crops), 1):
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
                record_num = len(self.crop_history) - len(visible_crops) + i
                text = (f"#{record_num}:({x1},{y1})→({x2},{y2})\n"
                       f"R²={r2:.4f}\n"
                       f"χ²ᵣ={chi2:.3f}\n"
                       f"S={score:.0f}/100")
            
            self.ax_record.text(0.05, y_pos, text,
                               transform=self.ax_record.transAxes,
                               fontsize=6, family='monospace',
                               verticalalignment='top',
                               color=color)
            y_pos -= record_height
            
            # 如果超出范围，停止显示
            if y_pos < 0.05:
                break
    
    def crop_image(self, x1, y1, x2, y2):
        """执行裁剪"""
        h, w = self.original_data.shape
        
        # 验证坐标
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        if x1 >= x2 or y2 >= y1:
            print("Error: Invalid coordinates! X1<X2, Y2<Y1")
            return False
        
        # 存储坐标
        self.crop_coords = (x1, y1, x2, y2)
        
        # 裁剪数据
        self.cropped_data = self.original_data[y2:y1, x1:x2]
        
        # 绘制红色框
        if self.rect:
            self.rect.remove()
        
        width = x2 - x1
        height = y1 - y2
        self.rect = Rectangle((x1, y2), width, height, 
                             linewidth=2, edgecolor='red', 
                             facecolor='none', linestyle='-')
        self.ax_image.add_patch(self.rect)
        
        # 更新histogram
        self.update_histogram(self.cropped_data)
        
        # 添加到历史记录
        self.crop_history.append({
            'coords': (x1, y1, x2, y2),
            'fit_score': self.current_fit_stats['fit_score'],
            'chi_square': self.current_fit_stats['reduced_chi_square'],
            'r_squared': self.current_fit_stats['r_squared']
        })
        
        # 更新record显示
        self.update_record_display()
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print(f"\n✓ Cropped: {self.cropped_data.shape[1]}x{self.cropped_data.shape[0]} pixels")
        print(f"  Region: [{x1}:{x2}, {y2}:{y1}]")
        print(f"  R² = {self.current_fit_stats['r_squared']:.4f}")
        print(f"  χ²ᵣ = {self.current_fit_stats['reduced_chi_square']:.3f}")
        print(f"  Score = {self.current_fit_stats['fit_score']:.1f}/100")
        
        return True
    
    def save_cropped(self):
        """保存裁剪后的文件"""
        if self.cropped_data is None or self.crop_coords is None:
            print("Error: No crop defined!")
            return False
        
        try:
            x1, y1, x2, y2 = self.crop_coords
            base_path = os.path.dirname(self.fits_file)
            base_name = os.path.basename(self.fits_file)
            name_part = os.path.splitext(base_name)[0]
            
            output_file = os.path.join(base_path, 
                                      f"{name_part}_crop_x{x1}-{x2}_y{y2}-{y1}.fits")
            
            # 更新header
            self.header['COMMENT'] = f'Cropped from {base_name}'
            self.header['CROPX1'] = (x1, 'Crop region X start')
            self.header['CROPX2'] = (x2, 'Crop region X end')
            self.header['CROPY1'] = (y2, 'Crop region Y start')
            self.header['CROPY2'] = (y1, 'Crop region Y end')
            self.header['CROPPED'] = (True, 'Image has been cropped')
            
            # 保存
            hdu = fits.PrimaryHDU(data=self.cropped_data, header=self.header)
            hdu.writeto(output_file, overwrite=True)
            
            print(f"\n✓ Saved: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False
    
    def run(self):
        """运行terminal交互循环"""
        print("\n" + "="*60)
        print("FITS Crop Tool - Terminal Mode")
        print("="*60)
        print(f"Image size: {self.original_data.shape[1]} x {self.original_data.shape[0]}")
        print("\nCommands:")
        print("  crop  - Crop a region")
        print("  list  - List all crop records")
        print("  save  - Save current crop")
        print("  quit  - Exit program")
        print("="*60 + "\n")
        
        while True:
            try:
                cmd = input("Enter command (crop/list/save/quit): ").strip().lower()
                
                if cmd == 'quit' or cmd == 'q':
                    print("Exiting...")
                    plt.close(self.fig)
                    break
                
                elif cmd == 'crop' or cmd == 'c':
                    # 获取previous坐标（如果存在）
                    if len(self.crop_history) > 1:  # 跳过Original记录
                        prev = self.crop_history[-1]['coords']
                        prev_str = f" (previous: {prev[0]}, {prev[1]}, {prev[2]}, {prev[3]})"
                    else:
                        prev_str = ""
                    
                    print(f"\nEnter crop coordinates{prev_str}:")
                    try:
                        x1_input = input("  Top-Left X: ").strip()
                        x1 = int(x1_input) if x1_input else (self.crop_history[-1]['coords'][0] if len(self.crop_history) > 1 else 0)
                        
                        y1_input = input("  Top-Left Y: ").strip()
                        y1 = int(y1_input) if y1_input else (self.crop_history[-1]['coords'][1] if len(self.crop_history) > 1 else self.original_data.shape[0])
                        
                        x2_input = input("  Bottom-Right X: ").strip()
                        x2 = int(x2_input) if x2_input else (self.crop_history[-1]['coords'][2] if len(self.crop_history) > 1 else self.original_data.shape[1])
                        
                        y2_input = input("  Bottom-Right Y: ").strip()
                        y2 = int(y2_input) if y2_input else (self.crop_history[-1]['coords'][3] if len(self.crop_history) > 1 else 0)
                        
                        confirm = input(f"\nConfirm crop [{x1},{y1}] to [{x2},{y2}]? (y/n): ").strip().lower()
                        if confirm == 'y' or confirm == 'yes':
                            self.crop_image(x1, y1, x2, y2)
                        else:
                            print("Crop cancelled.")
                    except ValueError:
                        print("Error: Invalid input! Please enter integers.")
                
                elif cmd == 'save' or cmd == 's':
                    self.save_cropped()
                
                elif cmd == 'list' or cmd == 'l':
                    # 显示所有crop记录
                    print("\n" + "="*60)
                    print(f"All Crop Records ({len(self.crop_history)} total):")
                    print("="*60)
                    for idx, record in enumerate(self.crop_history):
                        x1, y1, x2, y2 = record['coords']
                        score = record['fit_score']
                        chi2 = record['chi_square']
                        r2 = record['r_squared']
                        label = record.get('label', '')
                        
                        if label == 'Original':
                            print(f"\n[0] Original Image")
                            print(f"    Size: {x2} x {y1}")
                        else:
                            print(f"\n[{idx}] Crop: ({x1},{y1}) → ({x2},{y2})")
                            print(f"    Size: {x2-x1} x {y1-y2}")
                        
                        print(f"    R² = {r2:.4f}")
                        print(f"    χ²ᵣ = {chi2:.3f}")
                        print(f"    Score = {score:.1f}/100")
                    print("="*60 + "\n")
                
                else:
                    print("Unknown command. Use: crop, list, save, or quit")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                plt.close(self.fig)
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    # Get script directory and construct path to FITS file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    fits_file = os.path.join(project_root, 'Fits_Data', 'mosaic.fits')
    
    if not os.path.exists(fits_file):
        print(f"Error: FITS file not found at {fits_file}")
        return
    
    # Create and run the crop tool
    tool = FITSCropToolTerminal(fits_file)
    tool.run()


if __name__ == '__main__':
    main()
