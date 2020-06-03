import numpy as np
import cv2

picture_address = '1.jpg'


def spawn_grid_raster(x_pixel, y_pixel, x_period, y_period, grid_type='sine', square_wide_x=0.5, square_wide_y=0.5):
    """生成光栅图样
    x_pixel，y_pixel：x、y方向图片大小/像素
    x_period, y_period：x、y方向周期/像素
    type：样式，默认'sine'正弦光栅，'square'条纹光栅
    square_wide：如果是条纹光栅才可用，这里是指占空比，默认0.5"""
    ranges = [-int(x_pixel / 2), x_pixel - int(x_pixel / 2), -int(y_pixel / 2), y_pixel - int(y_pixel / 2)]
    x, y = np.meshgrid(np.arange(ranges[0], ranges[1], 1), np.arange(ranges[2], ranges[3], 1))
    if grid_type == 'sine':
        x_raster = 0.5 + 0.5 * np.cos(x * np.pi * 2 / x_period)
        y_raster = 0.5 + 0.5 * np.cos(y * np.pi * 2 / y_period)
        raster0 = np.array((x_raster, y_raster)).max(axis=0)
    elif grid_type == 'square':
        x_raster = np.cos(x * np.pi * 2 / x_period) > np.cos(square_wide_x * np.pi)
        y_raster = np.cos(y * np.pi * 2 / y_period) > np.cos(square_wide_y * np.pi)
        raster0 = x_raster | y_raster
    else:
        raster0 = np.random.random([y_pixel, x_pixel])
    return np.array(raster0, dtype='float')


def main():
    sin_raster = spawn_grid_raster(1000, 750, 100, 100)
    sqr_raster = spawn_grid_raster(200, 200, 15, 15, 'square', square_wide_x=0.2, square_wide_y=0.2)
    cv2.imshow('sqr_raster', sqr_raster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(picture_address, sqr_raster * 255)


if __name__ == '__main__':
    main()
