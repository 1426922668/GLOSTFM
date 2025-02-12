# coding=utf-8
#@Time : 2023/9/1 9:46
#@Author : 陈士泽
#@File :LST_preprocess.py
#@Software : PyCharm

from osgeo import osr
from utils import *
import h5py
import os


class MERSI_LST_preprocess:
    def __init__(self,folder_folder, folder_output, date_start = None, date_end = None, band_num = 5, left= "Z",right="H",up="8",down="H"):
        self.folder_folder = folder_folder # folder_folder为包含各日期MERSIhdf的文件夹的文件夹（2级文件夹）
        self.folder_temp = folder_folder + "_temp"  # 处理的中间数据，集中清除
        self.folder_output = folder_output
        self.date_start = str(date_start) # 预处理文件的时间范围 起始时间 例如 20201010 即2020年10月10日
        self.date_end = str(date_end) # 预处理文件的时间范围 终止时间 例如 20201030 即2020年10月30日
        self.dates = date_list(self.date_start,self.date_end)
        self.date_processing_date = self.date_start # 正处理的日期，初始值
        self.folder_temp_date = self.folder_temp + "/" + self.date_processing_date # 按日期划分的temp的子文件夹
        self.band_num = band_num # MERSI LST产品对应的波段 一般使用默认值 按实际数据更改
        self.left = left  # 按照FY3行列号指定特定区域(左侧列号) 生成LST图像 如果世界范围则按照默认值即可，处理时间会比较长
        self.right = right  # 按照FY3行列号指定特定区域(右侧列号) 生成LST图像 如果世界范围则按照默认值即可，处理时间会比较长
        self.up = up  # 按照FY3行列号指定特定区域(上侧行号) 生成LST图像 如果世界范围则按照默认值即可，处理时间会比较长
        self.down = down  # 按照FY3行列号指定特定区域(下侧行号) 生成LST图像 如果世界范围则按照默认值即可，处理时间会比较长

        if date_start == None:
            print("请输入预处理文件起始时间!")
        if date_end == None or date_end < date_start:
            print("请输入预处理文件起始时间!")
        mkdir(self.folder_temp)
        mkdir(self.folder_output)

    def date_update(self):  # 更新处理日期，处理下一景文件
        for self.date_processing_date in self.dates:
            self.folder_temp_date = self.folder_temp + "/" + self.date_processing_date  # 按日期划分的temp的子文件夹
            if not os.path.exists(self.folder_folder + "//" + self.date_processing_date):
                continue
            MERSI_LST_preprocess.HDF2TifFolder(self, self.folder_folder + "//" + self.date_processing_date,
                                               self.folder_temp_date, band_num=self.band_num)
            MERSI_LST_preprocess.MERSI_LST_mosaic(self)
        del_folder(self.folder_temp)   # 删除中间文件
        return self.folder_output


    def HDF2TifFolder(self, in_file, out_file, band_num):
        left_index, right_index, up_index, down_index = MERSI_LST_preprocess.FY3_row_col(col_str=self.left)[1],\
                                                        MERSI_LST_preprocess.FY3_row_col(col_str=self.right)[1],\
                                                        MERSI_LST_preprocess.FY3_row_col(row_str=self.up)[0],\
                                                        MERSI_LST_preprocess.FY3_row_col(row_str=self.down)[0]
        mkdir(out_file)
        tifNameList = sorted(os.listdir(in_file))
        for i in range(len(tifNameList)):
            file_row = tifNameList[i].split("_")[-9][0]
            file_col = tifNameList[i].split("_")[-9][2]

            file_row_index = MERSI_LST_preprocess.FY3_row_col(row_str=str(file_row))[0]
            file_col_index = MERSI_LST_preprocess.FY3_row_col(col_str=str(file_col))[1]

            if not (up_index <= file_row_index <= down_index and left_index <= file_col_index <= right_index):
                continue
            if os.path.splitext(tifNameList[i])[1].lower() == ".hdf":
                in_path = in_file + "/" + tifNameList[i]
                out_path = out_file + "/" + tifNameList[i][:-4] + ".tif"
                MERSI_LST_preprocess.HDF2Tif_LST_MERSI(in_path, out_path, band_num)
        print('Finish ' + "HDF2TifFolder: " + in_file)
        pass

    def MERSI_LST_mosaic(self):
        # 默认是生成世界范围的LST，如果初始化输入了指点区域行列号，则按着生成区域LST
        left_index, right_index, up_index, down_index = MERSI_LST_preprocess.FY3_row_col(col_str=self.left)[1],\
                                                        MERSI_LST_preprocess.FY3_row_col(col_str=self.right)[1],\
                                                        MERSI_LST_preprocess.FY3_row_col(row_str=self.up)[0],\
                                                        MERSI_LST_preprocess.FY3_row_col(row_str=self.down)[0]
        region_rows, region_cols = (down_index - up_index + 1) * 1000, (right_index-left_index + 1) * 1000
        res_x, res_y = 1113.19490793273, 1113.19490793273
        region_array = np.zeros((region_rows, region_cols))

        world_Xcoor = -20037508.342789244
        world_Ycoor = 10018754.171394622

        upleft_Xcoor = left_index * res_x * 1000 + world_Xcoor
        upleft_Ycoor = world_Ycoor - up_index * res_y * 1000

        tifNameList = sorted(os.listdir(self.folder_temp_date))

        for i in range(len(tifNameList)):
            if os.path.splitext(tifNameList[i])[1].lower() != ".tif":
                continue
            in_path = self.folder_temp_date + "/" + tifNameList[i]

            ds = image_open(in_path)
            band = ds.GetRasterBand(1)
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            data_band = band.ReadAsArray(0, 0, cols, rows)
            geotransform = ds.GetGeoTransform()

            file_row = tifNameList[i].split("_")[-9][0]
            file_col = tifNameList[i].split("_")[-9][2]

            file_row_index = MERSI_LST_preprocess.FY3_row_col(row_str=str(file_row))[0]
            file_col_index = MERSI_LST_preprocess.FY3_row_col(col_str=str(file_col))[1]

            array_row_index = (file_row_index - up_index) * 1000
            array_col_index = (file_col_index - left_index) * 1000

            region_array[array_row_index: array_row_index + rows, array_col_index: array_col_index + cols] = data_band

        region_array = region_array.astype(int)
        driver = gdal.GetDriverByName("GTiff")

        outds = driver.Create(self.folder_output+"//"+ self.date_processing_date + ".tif", region_cols, region_rows, 1, gdal.GDT_Int16)
        left_xx, left_yy = upleft_Xcoor, upleft_Ycoor

        outds.SetGeoTransform((left_xx, res_x, 0, left_yy, 0, -res_y))
        proj = 'PROJCS["World_Plate_Carree", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0],UNIT["Degree", 0.0174532925199433]], PROJECTION["Plate_Carree"], PARAMETER["False_Easting", 0.0], PARAMETER["False_Northing", 0.0], PARAMETER["Central_Meridian", 0.0], UNIT["Meter", 1.0]]'

        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(region_array)
        outds.FlushCache()
        print('Finish ' + "MERSI_mosaic: " + self.date_processing_date)
        pass

    @staticmethod
    def FY3_row_col(row_str="H", col_str="H"):
        col_list = ["Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "M", "N", "L", "K", "J", "I",
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H"]
        row_list = ["8", "7", "6", "5", "4", "3", "2", "1", "0", "9", "A", "B", "C", "D", "E", "F", "G", "H"]

        col_index = int(col_list.index(col_str))
        row_index = int(row_list.index(row_str))

        index_array = [row_index,col_index]
        # left_index = col_list.index(self.left)
        # right_index = col_list.index(self.right)
        # up_index = row_list.index(self.up)
        # down_index = row_list.index(self.down)
        # if left_index > right_index or up_index > down_index:
        #     print("请检查所输入的行列号是否正确！")
        return index_array

    @staticmethod
    def HDF2Tif_LST_MERSI(in_file, out_file, band_num):  # , prosrs_para):
        hdf_ds = h5py.File(in_file, "r")
        if type(hdf_ds.attrs['Left-Top X']) is np.ndarray:
            left_x = hdf_ds.attrs['Left-Top X'][0]
        else:
            left_x = float(hdf_ds.attrs['Left-Top X'])

        if type(hdf_ds.attrs['Left-Top Y']) is np.ndarray:
            left_y = hdf_ds.attrs['Left-Top Y'][0]
        else:
            left_y = float(hdf_ds.attrs['Left-Top Y'])

        if type(hdf_ds.attrs['Right-Bottom X']) is np.ndarray:
            right_x = hdf_ds.attrs['Right-Bottom X'][0]
        else:
            right_x = float(hdf_ds.attrs['Right-BottomX'])

        if type(hdf_ds.attrs['Right-Bottom Y']) is np.ndarray:
            right_y = hdf_ds.attrs['Right-Bottom Y'][0]
        else:
            right_y = float(hdf_ds.attrs['Right-Bottom Y'])

        n_name = list(hdf_ds.keys())[band_num]

        n_ds = hdf_ds[n_name]
        rows = n_ds.shape[0]
        cols = n_ds.shape[1]
        data = n_ds[()]
        driver = gdal.GetDriverByName("GTiff")
        outds = driver.Create(out_file, cols, rows, 1, gdal.GDT_Int16)

        # a = lonlat2geo(left_x, left_y)
        left_xx, left_yy = MERSI_LST_preprocess.lonlat2geo(float(left_x), float(left_y))
        right_xx, right_yy = MERSI_LST_preprocess.lonlat2geo(float(right_x), float(right_y))
        res_x = (right_xx - left_xx) / cols
        res_y = (right_yy - left_yy) / rows

        outds.SetGeoTransform(
            (left_xx, res_x, 0,
             left_yy, 0, res_y))
        proj = 'PROJCS["World_Plate_Carree", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0],UNIT["Degree", 0.0174532925199433]], PROJECTION["Plate_Carree"], PARAMETER["False_Easting", 0.0], PARAMETER["False_Northing", 0.0], PARAMETER["Central_Meridian", 0.0], UNIT["Meter", 1.0]]'
        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(data)
        pass

    @staticmethod
    def lonlat2geo(lon, lat):
        prosrs1 = 'PROJCS["World_Plate_Carree", GEOGCS["GCS_WGS_1984", DATUM["D_WGS_1984", SPHEROID["WGS_1984", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0],UNIT["Degree", 0.0174532925199433]], PROJECTION["Plate_Carree"], PARAMETER["False_Easting", 0.0], PARAMETER["False_Northing", 0.0], PARAMETER["Central_Meridian", 0.0], UNIT["Meter", 1.0]]'
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(prosrs1)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        return coords[:2]


if __name__ == '__main__':
    test1=MERSI_LST_preprocess("F:\data\FY-3D原始数据_第二批\FY3D-DailyLst-202306-07",20230601,20230801,left= "7",right="D",up="5",down="2")
    test1.date_update()