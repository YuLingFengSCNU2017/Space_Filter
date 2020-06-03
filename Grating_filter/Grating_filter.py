import cv2
import numpy as np
import skimage.util as skut
import matplotlib.pyplot as plt

picture_address = '2.jpg'
d_rad = 2
plot_all = 0  # bool


def create_filter(condition, x_pixel, y_pixel=None):  # x -> j, and y -> -i
    """返回一个滤波器。
    可通过的范围条件（表达式），x像素值，y像素值"""
    if y_pixel is None:
        y_pixel = x_pixel
    ranges = [-int(x_pixel / 2), x_pixel - int(x_pixel / 2), -int(y_pixel / 2), y_pixel - int(y_pixel / 2)]
    x, y = np.meshgrid(np.arange(ranges[0], ranges[1], 1), np.arange(ranges[2], ranges[3], 1))
    filter0 = np.ones([y_pixel, x_pixel]) * (eval(condition))
    return filter0


def draw_pics(window_name, pic):
    """多图绘图工具
    [图1窗口，图2窗口……],[图1，图2……]"""
    if type(window_name) == str:
        window_name, pic = [window_name], [pic]  # show only 1 picture
    for i in range(len(pic)):
        cv2.imshow(window_name[i], pic[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def catchpic(pic_name):
    """读取'pic_name'中的图片，返回灰度图和开根号的复振幅（倾角0）
    pic_name:图片的位置"""
    pic_gray = cv2.imread(pic_name, 0)
    pic_sqrt = np.sqrt(pic_gray)
    pic_gray = np.array(pic_gray / pic_gray.max(), dtype='float64')  # float64 & uint8
    pic_sqrt = np.array(pic_sqrt / pic_sqrt.max(), dtype='float64')  # float64 & uint8
    return pic_gray, pic_sqrt


def comppic(pic1, pic2, bins=20, max_value=1, get_hist_pic=False, title=''):
    """输入两张图片，输出两张图片的绝对差值的分布直方图以及期望值
    默认数量是20个区间
    默认区间是[0，1]"""
    pixel = min(pic1.shape)
    different = cv2.absdiff(pic1, pic2).reshape(np.prod(pic1.shape))
    if get_hist_pic:
        plt.hist(different, bins, (0, max_value), density=True)
        # bins显示有几个直方,normed是否对数据进行标准化
        plt.title('Distribution of %s filter errors' % title)
        plt.xlabel('Error after filtering')
        plt.ylabel('Distribution density of pixels')
        plt.show()
    return np.sum(different ** 2) / pixel ** 2


def main():
    # 读取图片并转化为灰度值
    pic2, pic1 = catchpic(picture_address)
    pic_shape_xy = [pic2.shape[1], pic2.shape[0]]
    pixel = min(pic_shape_xy)  # pixel_x = pixel_y
    # draw_pics('pic-gray', pic2)

    # 添加噪声
    pic2_gauss = skut.random_noise(pic2, 'gaussian', var=0.09)  # 高斯噪声方差
    pic2_salt = skut.random_noise(pic2, 's&p', amount=0.09)  # 椒盐噪声比例
    pic2_mix = skut.random_noise(pic2_salt, 'gaussian', var=0.09)  # 混合
    pics2 = [pic2, pic2_gauss, pic2_salt, pic2_mix]  # 下面要选择第几项来继续
    draw_pics(['0', '1-gauss', '2-salt', '3-mix'], pics2)
    noise_title = ['None', 'Gaussian', 'Salt-Pepper', 'Mixture']
    f_r, err, a_err = [], [], []  # now_filter_radius, its error, all error

    # num_of_pic = eval(input('num of (0,1,2,3) for (None, Gauss, Salt, Mix) :'))
    for num_of_pic in range(4):
        filter_radius = d_rad
        f_r, err = [], [],  # clear all
        while filter_radius < (pixel / 2):
            is_plot = filter_radius == d_rad \
                      or filter_radius + d_rad >= (pixel / 2) \
                      or filter_radius < (pixel / 4) <= filter_radius + d_rad

            # 构造滤波器filter0
            circle_filter = 'np.sqrt(x ** 2 + y ** 2) < %s' % filter_radius
            # square_filter = '(np.abs(x)<11)&(np.abs(y)<11)'
            filter0 = create_filter(circle_filter, pic_shape_xy[0], pic_shape_xy[1])
            if is_plot and plot_all:
                print(filter_radius)
                draw_pics('filter', filter0 * 0.7)  # 灰色背景，黑色滤波器

            # 从上面选一个，FFT，移频，滤波
            pic1_0 = np.sqrt(pics2[num_of_pic])  # 0,1,2,3
            fft_pic1 = np.fft.fft2(pic1_0)
            fft_pic1 = np.fft.fftshift(fft_pic1)
            fft_filter0_pic1 = fft_pic1 * filter0

            if is_plot and plot_all:  # FFT后的肉眼所见图样
                fft_pic2 = np.array(np.abs(fft_pic1) ** 2 / np.abs(fft_pic1).max(), dtype='float64')
                fft_pic2_filter0 = 0.5 * np.array(cv2.merge([filter0 * 0.99, filter0 * 0.99, filter0 * 0.99])
                                                  + cv2.merge([fft_pic2 * 0.8, fft_pic2 * 0.99, fft_pic2 * 0.7]),
                                                  dtype='float64')
                draw_pics('fft_pic2_filter0', fft_pic2_filter0)

            # 移频，ifft，绘结果图
            fft_filter0_pic1 = np.fft.ifftshift(fft_filter0_pic1)
            filter0_pic1 = np.fft.ifft2(fft_filter0_pic1)
            filter0_pic2 = np.abs(filter0_pic1) ** 2
            filter0_pic2 = np.array(filter0_pic2, dtype='float64')
            if is_plot and plot_all:
                draw_pics('pic_filter_finished', filter0_pic2)

            # 滤波后与原像素作对比
            if is_plot:
                comppic(pic2, filter0_pic2, get_hist_pic=True,
                        title='noise=%s, radius=%s' % (noise_title[num_of_pic], filter_radius))
            err.append(comppic(pic2, filter0_pic2))
            f_r.append(filter_radius)
            filter_radius += d_rad
        a_err.append(err)
    for num_of_pic in range(4):
        plt.plot(f_r, a_err[num_of_pic], label=noise_title[num_of_pic])
    plt.title('Filter MSE distribution under different noises')
    plt.xlabel('Filter radius')
    plt.ylabel('Mean-Square Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
