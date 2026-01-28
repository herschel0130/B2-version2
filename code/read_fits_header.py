"""
读取FITS文件的Header信息
用于快速查看天文图像的元数据和关键参数
"""

import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np


def read_and_display_header(fits_path: str, verbose: bool = False):
    """
    读取并显示FITS文件的header信息
    
    Args:
        fits_path: FITS文件路径
        verbose: 是否显示所有header信息
    """
    path = Path(fits_path)
    if not path.exists():
        print(f"❌ 文件不存在: {fits_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"📁 文件: {path.name}")
    print(f"📂 路径: {path.parent}")
    print(f"{'='*70}\n")
    
    with fits.open(fits_path) as hdul:
        # 显示FITS文件结构
        print("📊 FITS文件结构:")
        print("-" * 70)
        hdul.info()
        print()
        
        # 获取主HDU
        primary_hdu = hdul[0]
        header = primary_hdu.header
        data = primary_hdu.data
        
        # 显示图像基本信息
        if data is not None:
            print("🖼️  图像信息:")
            print("-" * 70)
            print(f"维度 (NAXIS):        {data.ndim}")
            print(f"形状 (Shape):        {data.shape}")
            print(f"数据类型:            {data.dtype}")
            
            # 统计信息
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                print(f"最小值:              {np.min(finite_data):.6g}")
                print(f"最大值:              {np.max(finite_data):.6g}")
                print(f"中位数:              {np.median(finite_data):.6g}")
                print(f"平均值:              {np.mean(finite_data):.6g}")
                print(f"标准差:              {np.std(finite_data):.6g}")
            print()
        
        # 显示关键天文参数
        print("🌟 关键天文参数:")
        print("-" * 70)
        
        # 重要的header关键字
        important_keys = {
            'MAGZPT': '星等零点 (Magnitude Zero Point)',
            'MAGZRR': '零点误差 (Zero Point Error)',
            'FILTER': '滤光片 (Filter)',
            'EXPTIME': '曝光时间 (秒)',
            'GAIN': '增益 (e-/ADU)',
            'RDNOISE': '读出噪声 (e-)',
            'TELESCOP': '望远镜 (Telescope)',
            'INSTRUME': '仪器 (Instrument)',
            'OBSERVER': '观测者 (Observer)',
            'OBJECT': '观测目标 (Object)',
            'DATE-OBS': '观测日期 (Date)',
            'RA': '赤经 (Right Ascension)',
            'DEC': '赤纬 (Declination)',
            'AIRMASS': '大气质量 (Airmass)',
            'PIXSCALE': '像元尺度 (arcsec/pixel)',
            'SEEING': '视宁度 (arcsec)',
        }
        
        found_any = False
        for key, description in important_keys.items():
            if key in header:
                value = header[key]
                print(f"{description:30s} = {value}")
                found_any = True
        
        if not found_any:
            print("⚠️  未找到标准的关键参数")
        print()
        
        # 如果是verbose模式，显示所有header
        if verbose:
            print("📋 完整Header信息:")
            print("-" * 70)
            print(repr(header))
            print()
        else:
            print("💡 提示: 使用 --verbose 参数查看完整header信息")
            print()
        
        # 显示header条目数量
        print(f"📝 Header共包含 {len(header)} 个关键字")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="读取并显示FITS文件的header信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python read_fits_header.py mosaic.fits
  python read_fits_header.py mosaic.fits --verbose
  python read_fits_header.py ../Astro/Fits_Data/mosaic.fits
        """
    )
    
    parser.add_argument(
        'fits_file',
        type=str,
        help='FITS文件路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示完整的header信息（包括所有关键字）'
    )
    
    args = parser.parse_args()
    
    try:
        read_and_display_header(args.fits_file, args.verbose)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
