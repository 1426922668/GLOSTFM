from config import *
import os
from tqdm import tqdm
import numpy as np
import cupy as cp
import rasterio
from utils import *


def linear_spectrum_blur(interpolate_array, window_size, edge_smooth=None):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    interpolate_array_sum = cp.zeros([len(interpolate_array), len(interpolate_array[0])]).astype(float)
    interpolate_array_weight = cp.zeros([len(interpolate_array), len(interpolate_array[0])]).astype(float)

    interpolate_array = np.pad(interpolate_array, int(window_size / 2))

    interpolate_array_01 = np.where(interpolate_array, 1, 0)

    interpolate_array = cp.array(interpolate_array)
    interpolate_array_01 = cp.array(interpolate_array_01)

    for i in tqdm(range(window_size)):
        for j in range(window_size):
            interpolate_array_sum += interpolate_array[i: len(interpolate_array) - window_size + i + (window_size % 2),
                                     j: len(interpolate_array[0]) - window_size + j + (window_size % 2)]
            interpolate_array_weight += interpolate_array_01[
                                        i: len(interpolate_array) - window_size + i + (window_size % 2),
                                        j: len(interpolate_array[0]) - window_size + j + (window_size % 2)]
    interpolate_array = None
    output_array = cp.divide(interpolate_array_sum, interpolate_array_weight)
    interpolate_array_sum, interpolate_array_01 = 0, 0
    output_array = cp.nan_to_num(output_array, nan=0)
    output_array = cp.asnumpy(output_array)
    interpolate_array_weight = cp.asnumpy(interpolate_array_weight)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return output_array, interpolate_array_weight


class RC:
    def __init__(self, compensation_ref_array, need_compensation_array):
        self.compensation_ref_array = compensation_ref_array
        self.need_compensation_array = need_compensation_array

    def residual_compensation(self, window_size=50, shrinkage=40):
        residual_array = RC.residual_calculate(self)
        self.need_compensation_array = np.where(self.need_compensation_array,
                                                self.need_compensation_array + RC.mi_san(residual_array, shrinkage), 0)
        residual_array = RC.residual_calculate(self)

        residual_array_blur = RC.gauss_compensation(residual_array, window_size)

        self.need_compensation_array = np.where(self.need_compensation_array,
                                                self.need_compensation_array + residual_array_blur, 0)

        compensated_array = np.where(self.compensation_ref_array, self.compensation_ref_array,
                                     self.need_compensation_array)
        return compensated_array

    def residual_calculate(self):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        compensation_ref_array = cp.array(self.compensation_ref_array).astype(cp.float16)
        need_compensation_array = cp.array(self.need_compensation_array).astype(cp.float16)
        need_compensation_array = cp.where(compensation_ref_array, 0, need_compensation_array)

        compensation_ref_array = cp.pad(compensation_ref_array, 1)

        residual_array = cp.where(need_compensation_array * compensation_ref_array[:-2, 1:-1],
                                  compensation_ref_array[:-2, 1:-1] - need_compensation_array, 0)
        residual_array_num = cp.where(need_compensation_array * compensation_ref_array[:-2, 1:-1], 1, 0)

        residual_array += cp.where(need_compensation_array * compensation_ref_array[2:, 1:-1],
                                   compensation_ref_array[2:, 1:-1] - need_compensation_array, 0)
        residual_array_num += cp.where(need_compensation_array * compensation_ref_array[2:, 1:-1], 1, 0)

        residual_array += cp.where(need_compensation_array * compensation_ref_array[1:-1, :-2],
                                   compensation_ref_array[1:-1, :-2] - need_compensation_array, 0)
        residual_array_num += cp.where(need_compensation_array * compensation_ref_array[1:-1, :-2], 1, 0)

        residual_array += cp.where(need_compensation_array * compensation_ref_array[1:-1, 2:],
                                   compensation_ref_array[1:-1, 2:] - need_compensation_array, 0)
        residual_array_num += cp.where(need_compensation_array * compensation_ref_array[1:-1, 2:], 1, 0)

        residual_array = cp.divide(residual_array, residual_array_num)
        residual_array_num = None
        residual_array = cp.nan_to_num(residual_array, nan=0)

        residual_array = cp.asnumpy(residual_array)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return residual_array

    @staticmethod
    def mi_san(array, Shrinkage=40):
        col = len(array[0])
        row = len(array)
        array = RC.resize_mean(array, Shrinkage)
        array = RC.interpolation(array)
        array = cv2.resize(np.float32(array), (col, row), interpolation=cv2.INTER_LINEAR)
        return array

    @staticmethod
    def gauss_compensation(residual_array, window_size):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        window_size += (window_size + 1) % 2
        unit_kernel = np.zeros([window_size, window_size])
        unit_kernel[int(window_size / 2), int(window_size / 2)] = 1
        unit_kernel = cv2.GaussianBlur(unit_kernel, (window_size, window_size), 0, 0)

        residual_array_numer = cp.zeros([len(residual_array), len(residual_array[0])]).astype(cp.float16)
        residual_array_denom = cp.zeros([len(residual_array), len(residual_array[0])]).astype(cp.float16)
        residual_array = np.pad(residual_array, int(window_size / 2))
        residual_array = cp.array(residual_array).astype(cp.float16)

        for i in tqdm(range(window_size)):
            for j in range(window_size):
                distance_1 = ((int(window_size / 2) - abs(i - int(window_size / 2)) + 1) ** 2 + (
                        int(window_size / 2) - abs(j - int(window_size / 2)) + 1) ** 2) ** 0.5

                residual_array_shk = unit_kernel[i, j] * residual_array[
                                                         i: len(residual_array) - window_size + i + (window_size % 2),
                                                         j: len(residual_array[0]) - window_size + j + (
                                                                 window_size % 2)]

                residual_array_numer += cp.abs(10 ** (distance_1 - int(window_size / 2))) * residual_array_shk
                residual_array_denom += cp.abs(10 ** (distance_1 - int(window_size / 2))) * cp.where(residual_array_shk,
                                                                                                     1, 0)

        residual_array_blur = cp.divide(residual_array_numer, residual_array_denom)
        residual_array_blur = cp.nan_to_num(residual_array_blur, nan=0)
        residual_array_blur = cp.asnumpy(residual_array_blur)
        residual_array_blur /= unit_kernel[int(window_size / 2), int(window_size / 2)]

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return residual_array_blur

    @staticmethod
    def resize_mean(need_resize_array, Shrinkage):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        need_resize_array = cp.array(need_resize_array).astype(cp.float16)
        tmp_s = cp.zeros([int(len(need_resize_array) / Shrinkage), int(len(need_resize_array[0]) / Shrinkage)]).astype(
            cp.float16)
        tmp_w = cp.zeros([int(len(need_resize_array) / Shrinkage), int(len(need_resize_array[0]) / Shrinkage)]).astype(
            cp.float16)
        need_resize_array_01 = cp.where(need_resize_array, 1, 0) + eps
        for i in range(Shrinkage):
            for j in range(Shrinkage):
                tmp_s += need_resize_array[i::Shrinkage, j::Shrinkage][:len(tmp_s),:len(tmp_s)]
                tmp_w += need_resize_array_01[i::Shrinkage, j::Shrinkage][:len(tmp_w),:len(tmp_w)]
        tmp_s /= tmp_w
        tmp_s = cp.asnumpy(tmp_s)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return tmp_s

    @staticmethod
    def interpolation(need_interp_array, window_size=3):
        row = len(need_interp_array)
        col = len(need_interp_array[0])

        tmp = need_interp_array
        tmp = np.pad(tmp, window_size // 2)
        tmp = cp.array(tmp).astype(cp.float16)

        for i in tqdm(range(row)):
            for j in range(col):
                if abs(tmp[i + window_size // 2, j + window_size // 2]) > eps:
                    continue
                else:
                    tmp_s, tmp_w = 0, eps
                    for m in range(window_size):
                        for n in range(window_size):
                            if abs(tmp[i + m, j + n]) > eps:
                                tmp_s += tmp[i + m, j + n]
                                tmp_w += 1
                    tmp[i + window_size // 2, j + window_size // 2] = tmp_s / tmp_w

        for i in tqdm(range(row - 1, -1, -1)):
            for j in range(col - 1, -1, -1):
                if abs(tmp[i + window_size // 2, j + window_size // 2]) > eps:
                    continue
                else:
                    tmp_s, tmp_w = 0, eps
                    for m in range(window_size):
                        for n in range(window_size):
                            if abs(tmp[i + m, j + n]) > eps:
                                tmp_s += tmp[i + m, j + n]
                                tmp_w += 1
                    tmp[i + window_size // 2, j + window_size // 2] = tmp_s / tmp_w
        tmp = tmp[window_size // 2:window_size // 2 + row, window_size // 2:window_size // 2 + col]
        tmp = cp.asnumpy(tmp)
        return tmp


def num2str(num_int: int):
    if num_int < 10:
        num_str = "0" + str(num_int)
    else:
        num_str = str(num_int)
    return num_str


def date_list(start_date, end_date):
    start_date, end_date = str(start_date), str(end_date)
    print("列表日期方向{}->{}".format(start_date, end_date))

    reverse_flag = False
    if start_date > end_date:
        reverse_flag = True
        temp_date = start_date
        start_date = end_date
        end_date = temp_date
        # print("请终止程序，检查起始与终止日期！")

    list_date = [start_date]

    pros_date = start_date
    while pros_date < end_date:
        # pros_date update (day, month, year)
        num_day = int(pros_date[-2:])  # day
        if num_day < 31:
            num_day += 1
            pros_date = pros_date[:-2] + num2str(num_day)
        else:
            pros_date = pros_date[:-2] + "01"
            num_month = int(pros_date[-4:-2])  # month
            if num_month < 12:
                num_month += 1
                pros_date = pros_date[:4] + num2str(num_month) + pros_date[-2:]
            else:
                pros_date = pros_date[:4] + "01" + pros_date[-2:]
                num_year = int(pros_date[:4])  # year
                num_year += 1
                pros_date = str(num_year) + pros_date[4:]
        list_date.append(pros_date)

        list_date.sort(reverse=reverse_flag)

    return list_date


class ArrayIndex:
    def __init__(self, array_input, nodata_value=0):
        self.array_input = array_input
        self.array_01 = np.where(self.array_input != nodata_value, 1, 0)
        self.sum_01 = np.sum(self.array_01) + eps

    def ME(self):
        array_ME = np.sum(self.array_input * self.array_01) / self.sum_01  # Mean Error
        return array_ME

    def MAE(self):
        array_MAE = np.sum(abs(self.array_input)) / self.sum_01  # Mean Average Error
        return array_MAE

    def RMSE(self):
        array_RMSE = (np.sum(self.array_input ** 2) / self.sum_01) ** 0.5  # Root Mean Square Error
        return array_RMSE

    def Std(self):
        array_std = (np.sum(((self.array_input - self.ME()) * self.array_01) ** 2) / self.sum_01) ** 0.5
        return array_std


class ReadDataToArray:
    def __init__(self, DataFile):
        self.file = DataFile
        self.name = DataFile.split('\\')[-1]
        self.suffix = DataFile.split('.')[-1]

    def auto_read(self):
        if self.name == r"":
            pass

    def tif_bandmath(self, slope_img=1, intercept_img=0, data_range=None, background=0, band_num=1):
        # 1.open
        tif_img = rasterio.open(self.file).read(band_num)
        array_img = np.array(tif_img, dtype=float)
        # 2.unit
        array_img *= slope_img
        array_img += intercept_img
        # 3.background
        if data_range:
            array_img = np.where((data_range[0] < array_img) * (array_img < data_range[1]), array_img, background)
        print("Read successfully:" + self.name)

        return array_img


class TextureUpdate:
    def __init__(self, fusion_folder, ref_row, ref_col, feature_type="texture"):
        self.file = fusion_folder + "/background/" + feature_type + ".tif"
        self.coverage_file = fusion_folder + "/background/" + feature_type + "_coverage.tif"
        self.coverDate_file = fusion_folder + "/background/" + feature_type + "_coverDate.tif"

        if os.path.exists(self.file):
            self.history = rasterio.open(self.file).read(1)
        else:
            self.history = np.zeros([ref_row, ref_col], dtype=float)

        if os.path.exists(self.coverage_file):
            self.coverage_history = rasterio.open(self.coverage_file).read(1)
        else:
            self.coverage_history = np.zeros([ref_row, ref_col], dtype=int)

        if os.path.exists(self.coverDate_file):
            self.coverDate_history = rasterio.open(self.coverDate_file).read(1)
        else:
            self.coverDate_history = np.zeros([ref_row, ref_col], dtype=int)

    def texture_update(self, hs_img_array, window_size: int, update_date: int):
        # texture extract
        print("linear_spectrum_blur of texture_update...")
        hs_img_blur, hs_array_coverage = linear_spectrum_blur(hs_img_array, window_size=blur_windowSize_FL)
        hs_img_texture = np.where(hs_img_array, hs_img_array - hs_img_blur, 0)

        # Standard value calculate
        texture_history_std = ArrayIndex(self.history, 999).Std()
        if not (abs(texture_history_std) < 5 and abs(texture_history_std) != 0):
            texture_history_std = texture_std_limit_p
        print("texture std value = " + str(texture_history_std))

        # Texture Filtering
        hs_texture_filter_01 = np.where(abs(hs_img_texture) <= update_alpha * abs(self.history), True, False)
        hs_texture_filter_01 = np.logical_or(hs_texture_filter_01,
                                             np.where(abs(hs_img_texture) <= 3 * texture_history_std, True, False))
        hs_texture_filter_01 = np.logical_or(hs_texture_filter_01,
                                             np.where(self.history == 0, True, False))
        hs_texture_filter_01 = np.logical_and(hs_texture_filter_01,
                                              np.where((abs(hs_img_texture) != 0) * (abs(hs_img_texture) < 15), True,
                                                       False))

        # Coverage Filtering
        hs_coverage_filter_01 = np.where(hs_array_coverage > self.coverage_history, True, False)
        hs_coverage_filter_01 = np.logical_or(hs_coverage_filter_01,
                                              np.where(hs_array_coverage >= ((window_size ** 2) * 0.5), True,
                                                       False))

        # filter synthesis
        hs_img_texture *= np.logical_and(hs_texture_filter_01, hs_coverage_filter_01)

        # update to array
        # self.history = np.where(hs_img_texture, hs_img_texture, self.history)
        self.history = RC(hs_img_texture, self.history).residual_compensation()
        self.coverage_history = np.where(hs_img_texture, hs_array_coverage, self.coverage_history)
        self.coverDate_history = np.where(hs_img_texture, update_date, self.coverDate_history)


class SDIUpdate(TextureUpdate):
    def __init__(self, fusion_folder, ref_row, ref_col):
        super(SDIUpdate, self).__init__(fusion_folder, ref_row, ref_col, feature_type="SDI")

    def sdi_update(self, hs_img_array, ls_img_array, update_date: int):
        print("linear_spectrum_blur1 of sdi_update...")
        hs_img_blur, _ = linear_spectrum_blur(hs_img_array, window_size=blur_windowSize_FL)
        print("linear_spectrum_blur2 of sdi_update...")
        ls_img_blur, _ = linear_spectrum_blur(ls_img_array, window_size=blur_windowSize_CL)
        HS_LS_SDI = hs_img_blur - ls_img_blur
        HS_LS_SDI = np.where(hs_img_array * ls_img_array, HS_LS_SDI, 0)

        # Standard value calculate
        sdi_history_std = ArrayIndex(self.history, 0).Std()
        if not (abs(sdi_history_std) < 10 and abs(sdi_history_std) != 0):
            sdi_history_std = sdi_std_limit_p
        print("sdi std value = " + str(sdi_history_std))

        # HS_LS_SDI = np.where((abs(HS_LS_SDI) <= 3 * sdi_history_std) * HS_LS_SDI, HS_LS_SDI, 0)
        HS_LS_SDI = np.where((abs(HS_LS_SDI) <= 3 * sdi_std_limit_p) * HS_LS_SDI, HS_LS_SDI, 0)

        # print("SDI residual composition...")
        # self.history = RC(HS_LS_SDI, self.history).residual_compensation(window_size=RC_F_window_size, shrinkage=RC_F_shrinkage)
        # self.history = np.where(self.history > 20 , 20, self.history) # 限幅
        # self.history = np.where(self.history < -20, -20, self.history)
        self.history = np.where(HS_LS_SDI, HS_LS_SDI, self.history)
        self.coverDate_history = np.where(HS_LS_SDI, update_date, self.coverDate_history)


class UpdatesAndCompositionAndInjection:
    def __init__(self, fusion_folder, HS_img_folder, LS_img_folder, ref_HS_tif):
        self.fusion_folder = fusion_folder
        self.HS_img_folder = HS_img_folder
        self.LS_img_folder = LS_img_folder
        mkdir(fusion_folder + "/background")

        self.update_startDate = None
        self.update_endDate = None
        self.update_skipDate = None

        self.LS_startDate = None
        self.LS_endDate = None
        self.composition_rule = None

        self.ref_data = gdal.Open(ref_HS_tif)

        self.ref_row = self.ref_data.RasterYSize
        self.ref_col = self.ref_data.RasterXSize

        # read texture history data
        self.texture_data = TextureUpdate(self.fusion_folder, self.ref_row, self.ref_col)
        self.SDI_data = SDIUpdate(self.fusion_folder, self.ref_row, self.ref_col)

    def features_update(self, update_startDate: str, update_endDate: str, update_skipDate: str):
        self.update_startDate = update_startDate
        self.update_endDate = update_endDate
        self.update_skipDate = update_skipDate

        dates_list = date_list(update_startDate, update_endDate)

        for i in tqdm(dates_list):
            if i == update_skipDate:
                continue
            HS_img_file = self.HS_img_folder + '/' + HS_img_nameHeadend + i + HS_img_nameBackend
            LS_img_file = self.LS_img_folder + '/' + LS_img_nameHeadend + i + LS_img_nameBackend

            if os.path.exists(HS_img_file):

                # read HS data
                HS_img_array = ReadDataToArray(HS_img_file).tif_bandmath(slope_img=HS_Slope, intercept_img=HS_intercept,
                                                                         data_range=[
                                                                             HS_minValue * HS_Slope + HS_intercept,
                                                                             HS_maxValue])
                # texture update
                self.texture_data.texture_update(hs_img_array=HS_img_array, window_size=blur_windowSize_FL,
                                                 update_date=int(i))

                if os.path.exists(LS_img_file):
                    LS_img_array = ReadDataToArray(LS_img_file).tif_bandmath(slope_img=LS_Slope,
                                                                             intercept_img=LS_intercept,
                                                                             data_range=[
                                                                                 LS_minValue * LS_Slope + LS_intercept,
                                                                                 LS_maxValue])

                    LS_img_array = cv2.resize(LS_img_array, (self.ref_col, self.ref_row),
                                              interpolation=cv2.INTER_NEAREST)

                    self.SDI_data.sdi_update(HS_img_array, LS_img_array, int(i))
            else:
                print(HS_img_file," not exist.")

        # save data as tif
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    self.texture_data.file, self.texture_data.history, gdal.GDT_Float32)
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    self.texture_data.coverage_file, self.texture_data.coverage_history, gdal.GDT_UInt16)
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    self.texture_data.coverDate_file, self.texture_data.coverDate_history, gdal.GDT_UInt32)

        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    self.SDI_data.file, self.SDI_data.history, gdal.GDT_Float32)
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    self.SDI_data.coverDate_file, self.SDI_data.coverDate_history, gdal.GDT_UInt32)

    def composition(self, LS_startDate, LS_endDate, composition_rule="Last"):
        self.LS_startDate = LS_startDate
        self.LS_endDate = LS_endDate
        self.composition_rule = composition_rule

        dates_list = date_list(LS_startDate, LS_endDate)

        flag, output_img, output_cunt = 0, 0, 0
        print("multi-days composition...")
        for i in tqdm(dates_list):
            LS_img_tif = self.LS_img_folder + '\\' + LS_img_nameHeadend + i + LS_img_nameBackend
            if not os.path.exists(LS_img_tif):
                continue
            LS_img_array = ReadDataToArray(LS_img_tif).tif_bandmath(slope_img=LS_Slope, intercept_img=LS_intercept,
                                                                    data_range=[LS_minValue * LS_Slope + LS_intercept,
                                                                                LS_maxValue])

            LS_img_array = cv2.resize(np.float32(LS_img_array),
                                      (int(self.ref_col / resolution_ratio), int(self.ref_row / resolution_ratio)),
                                      interpolation=cv2.INTER_NEAREST)
            LS_img_array = cp.array(LS_img_array)

            if flag == 0:
                # output_img = cp.zeros([len(LS_img_array), len(LS_img_array[0])]).astype(cp.float16)
                # output_cunt = cp.zeros([len(LS_img_array), len(LS_img_array[0])]).astype(cp.int16)
                output_img = cp.zeros([len(LS_img_array), len(LS_img_array[0])]).astype(cp.float16)
                output_cunt = cp.zeros([len(LS_img_array), len(LS_img_array[0])]).astype(cp.int16)
                flag += 1
            output_cunt += cp.where(LS_img_array, 1, 0)

            if composition_rule == "Max":
                output_img = cp.where(LS_img_array > output_img, LS_img_array, output_img)
            elif composition_rule == "Last":
                # output_img = cp.where(LS_img_array, LS_img_array, output_img)  # 最后一个覆盖运算符
                output_img = cp.array(RC(cp.asnumpy(LS_img_array), cp.asnumpy(output_img)).residual_compensation(
                    window_size=RC_C_window_size, shrinkage=RC_C_shrinkage))
            elif composition_rule == "First":
                # output_img = cp.where(output_img, output_img, LS_img_array)  # 第一个覆盖运算符
                output_img = cp.array(RC(cp.asnumpy(output_img), cp.asnumpy(LS_img_array)).residual_compensation(
                    window_size=RC_C_window_size, shrinkage=RC_C_shrinkage))
            elif composition_rule == "Mean":
                output_img += LS_img_array
            elif composition_rule == "Min":
                Min_rule1 = cp.where(LS_img_array < output_img, True, False)
                Min_rule2 = cp.where(LS_img_array != 0, True, False)
                Min_rule3 = cp.where(output_img == 0, True, False)
                Min_rule = cp.logical_and(Min_rule1, Min_rule2)
                Min_rule = cp.logical_or(Min_rule, Min_rule3)
                output_img = cp.where(Min_rule, LS_img_array, output_img)

        # 样本数量过少点位置，将直接设置为缺失，避免得到相对总体不准确的平均值
        output_img *= cp.where(output_cunt > (0.05 * len(dates_list)), 1, 0)

        if composition_rule == "Mean":
            output_img /= output_cunt
            output_cunt = 0
            output_img = cp.where(output_img < 40000, output_img, 0)

        LS_img_array = 0

        output_img = cp.asnumpy(output_img)

        output_img = cv2.resize(np.float32(output_img), (self.ref_col, self.ref_row), interpolation=cv2.INTER_NEAREST)

        if MWRI_background_switch:
            MWRI_background_img = rasterio.open(MWRI_background).read(1)
            MWRI_background_img = cv2.resize(np.float32(MWRI_background_img), (self.ref_col, self.ref_row),
                                             interpolation=cv2.INTER_NEAREST)
            output_img = RC(output_img, MWRI_background_img).residual_compensation(window_size=RC_F_window_size,
                                                                                   shrinkage=RC_F_shrinkage)
            output_img = np.where((output_img < outputValueMax) * (output_img > outputValueMin), output_img, MWRI_background_img)  # 限幅
            gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                        MWRI_background, output_img, gdal.GDT_Float32)  # MWRI背景场更新

        LS_composition_savefile = self.fusion_folder + "/" + "LS_" + self.LS_startDate + "_" + self.LS_endDate + "_" \
                                  + self.composition_rule + "_" + "_LST.tif"
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    LS_composition_savefile, output_img, gdal.GDT_Float32)

        return output_img

    def injection(self, ls_prediction_array, blur_windowSize_cl, include_sdi=True):
        output_img, _ = linear_spectrum_blur(ls_prediction_array, window_size=blur_windowSize_CL)

        HS_texture = self.texture_data.history
        HS_texture = np.where(abs(HS_texture > 30), 0, HS_texture)

        output_img = np.where(output_img * HS_texture, output_img + HS_texture, 0)

        if include_sdi:
            HS_LS_SDI = self.SDI_data.history
            # HS_LS_SDI *= np.where(abs(HS_LS_SDI) < 30, 1, 0)
            if output_sdi_blur_switch:
                HS_LS_SDI, _ = linear_spectrum_blur(self.SDI_data.history, window_size=blur_windowSize_CL)
            else:
                HS_LS_SDI = np.where(abs(self.SDI_data.history) < 30, self.SDI_data.history, 0)

            output_img = np.where(output_img * HS_LS_SDI, output_img + HS_LS_SDI, 0)

        output_img = np.where(output_img > outputValueMin, output_img, 0)
        output_img = np.where(output_img < outputValueMax, output_img, 0)

        if boundary_switch:
            boundary_img = rasterio.open(boundary_file).read(1)
            boundary_img = cv2.resize(np.float32(boundary_img), (self.ref_col, self.ref_row),
                                      interpolation=cv2.INTER_NEAREST)
            output_img = np.where(boundary_img == boundary_country, output_img, 0)
            output_img = np.where(boundary_img != boundary_water, output_img, 0)

        HS_fusion_savefile = self.fusion_folder + "/" + "HS_" + self.LS_startDate + "_" + self.LS_endDate + "_" \
                             + self.composition_rule + "_" + str(include_sdi) + "_LST.tif"
        gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                    HS_fusion_savefile, output_img, gdal.GDT_Float32)


class Verify:
    def __init__(self, sstfm_hs_tif, val_hs_tif):
        self.SSTFM_hs_tif = sstfm_hs_tif
        self.val_hs_tif = val_hs_tif

        self.SSTFM_hs_array = ReadDataToArray(sstfm_hs_tif).tif_bandmath(1, 0)

        self.val_hs_array = ReadDataToArray(val_hs_tif).tif_bandmath(slope_img=HS_Slope, intercept_img=HS_intercept,
                                                                     data_range=[
                                                                         HS_minValue * HS_Slope + HS_intercept,
                                                                         HS_maxValue])

    def error_index(self, generate_error_tif=False):
        error_array = self.SSTFM_hs_array - self.val_hs_array
        error_array *= np.where(self.SSTFM_hs_array * self.val_hs_array, 1, 0)
        error_cls = ArrayIndex(error_array)

        print("Error_ME:", error_cls.ME())
        print("Error_MAE:", error_cls.MAE())
        print("Error_RMSE:", error_cls.RMSE())

        self.ref_data = gdal.Open(ref_HS_tif)
        if generate_error_tif:
            gdal_writer(self.ref_data.GetProjection(), self.ref_data.GetGeoTransform(),
                        error_savefile, error_array, gdal.GDT_Float32)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    test1 = UpdatesAndCompositionAndInjection(fusion_folder, HS_img_folder, LS_img_folder, ref_HS_tif)
    test1.features_update(update_startDate, update_endDate, update_skipDate)
    LS_prediction_array = test1.composition(predict_startDate, predict_endDate, predict_composition_rule)
    test1.injection(LS_prediction_array, blur_windowSize_CL)

    end_time = datetime.datetime.now()
    print("Time consumed:", end_time - start_time)
