# coding=utf-8
# @Time : 2023/9/1 9:46
# @Author : 陈士泽
# @File :LST_preprocess.py
# @Software : PyCharm

from osgeo import osr
from utils import *
import h5py
from tqdm import tqdm


class MWRI_LST_preprocess:
    def __init__(self, folder, folder_output, MERSI_ref_tif, band_num=13):
        self.folder = folder  # MWRI原始数据文件夹路径，作为输入
        self.folder_temp = folder + "_temp"
        self.folder_output = folder_output  # 预处理完毕得MWRI数据，作为输出
        self.MERSI_ref_tif = MERSI_ref_tif  # 将MWRI投影坐标系与空间分辨率统一于MERSI，需要提供一张MERSI区域图片作为参考
        self.band_num = band_num  # 13与10 分别为升轨与降轨温度
        mkdir(self.folder_temp)
        mkdir(self.folder_output)

    def date_update(self):  # 更新处理日期，处理下一景文件
        """
        MWRI预处理包含两个部分：1.hdf图像转tif,基于hdf2tif函数；2.将MWRI统一到MERSI投影系；3.对MWRI图像的空间校正。
        """
        tifNameList = sorted(os.listdir(self.folder))
        for i in range(len(tifNameList)):
            if os.path.splitext(tifNameList[i])[1].lower() == ".hdf":
                in_path = self.folder + "/" + tifNameList[i]
                temp_path = self.folder_temp + "/" + tifNameList[i][:-4] + ".tif"
                out_path = self.folder_output + "/" + tifNameList[i][:-4] + ".tif"
                MWRI_LST_preprocess.hdf2tif(self, in_path, temp_path)  # 1.hdf图像转tif
                MWRI_LST_preprocess.geo2geo(self, temp_path, out_path)  # 2.将MWRI统一到MERSI投影系
        del_folder(self.folder_temp)   # 删除中间文件
        return self.folder_output

    def hdf2tif(self, in_path, out_path):
        band_num = self.band_num
        hdf_ds = h5py.File(in_path, "r")
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
        data = n_ds[()].astype(int)
        driver = gdal.GetDriverByName("GTiff")
        outds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Int16)

        # a = lonlat2geo(left_x, left_y)
        left_xx, left_yy = MWRI_LST_preprocess.lonlat2geo(float(left_x), float(left_y))
        right_xx, right_yy = MWRI_LST_preprocess.lonlat2geo(float(right_x), float(right_y))
        res_x = (right_xx - left_xx) / cols
        res_y = (right_yy - left_yy) / rows

        outds.SetGeoTransform(
            (left_xx, res_x, 0,
             left_yy, 0, res_y))

        proj = 'PROJCS["NSIDC_EASE_Grid_Global",GEOGCS["GCS_Sphere_International_1924_Authalic",' \
               'DATUM["D_Sphere_International_1924_Authalic",SPHEROID["Sphere_International_1924_Authalic",6371228.0,' \
               '0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION[' \
               '"Cylindrical_Equal_Area"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],' \
               'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",30.0],UNIT["Meter",1.0]] '
        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(data)
        pass

    def geo2geo(self, MWRI_original_file, out_path):
        start_time = datetime.datetime.now()

        MWRI_data = image_open(MWRI_original_file)
        MWRI_cols = MWRI_data.RasterXSize
        MWRI_rows = MWRI_data.RasterYSize
        MWRI_geotransform = MWRI_data.GetGeoTransform()
        MWRI_originX = MWRI_geotransform[0]
        MWRI_originY = MWRI_geotransform[3]
        MWRI_pixelWidth = MWRI_geotransform[1]
        MWRI_pixelHeight = MWRI_geotransform[5]
        MWRI_band = MWRI_data.GetRasterBand(1).ReadAsArray(0, 0, MWRI_cols, MWRI_rows).astype(np.float32)

        MWRI_x_vector = np.arange(MWRI_originX, MWRI_originX + MWRI_pixelWidth * MWRI_cols, MWRI_pixelWidth)
        MWRI_y_vector = np.arange(MWRI_originY, MWRI_originY + MWRI_pixelHeight * MWRI_rows, MWRI_pixelHeight)
        print("MWRI data OK!")
        print(datetime.datetime.now() - start_time)

        #########################
        proj_npy_x = os.path.abspath('.') + r"\MWRI_Lat_vector.npy"
        proj_npy_y = os.path.abspath('.') + r"\MWRI_Lon_vector.npy"
        if os.path.exists(proj_npy_x):
            MWRI_x_proj_vector = np.load(proj_npy_x)
            MWRI_y_proj_vector = np.load(proj_npy_y)
            print("读取npy成功")
        else:
            MWRI_y_proj_vector = map(MWRI_LST_preprocess.point_transform_y, MWRI_y_vector)
            MWRI_y_proj_vector = np.array(list(MWRI_y_proj_vector))

            MWRI_x_proj_vector = map(MWRI_LST_preprocess.point_transform_x, MWRI_x_vector)
            MWRI_x_proj_vector = np.array(list(MWRI_x_proj_vector))

            np.save(proj_npy_x, MWRI_x_proj_vector)
            np.save(proj_npy_y, MWRI_y_proj_vector)

        print("MWRI geo2geo_proj OK!")
        print(datetime.datetime.now() - start_time)

        #########################
        MERSI_data = image_open(self.MERSI_ref_tif)
        MERSI_cols = MERSI_data.RasterXSize
        MERSI_rows = MERSI_data.RasterYSize
        MERSI_geotransform = MERSI_data.GetGeoTransform()
        MERSI_originX = MERSI_geotransform[0]
        MERSI_originY = MERSI_geotransform[3]
        MERSI_pixelWidth = MERSI_geotransform[1]
        MERSI_pixelHeight = MERSI_geotransform[5]

        MERSI_x_vector = np.arange(MERSI_originX, MERSI_originX + MERSI_pixelWidth * MERSI_cols, MERSI_pixelWidth)
        MERSI_y_vector = np.arange(MERSI_originY, MERSI_originY + MERSI_pixelHeight * MERSI_rows, MERSI_pixelHeight)
        print("MERSI data OK!")

        index_set1 = 0
        spatial_bias = 1  # 3.对MWRI图像的空间校正
        y_index = np.zeros(MERSI_rows).astype(np.int16)
        for i in tqdm(range(MERSI_rows)):
            if index_set1 == (MWRI_rows - 1):
                pass
            else:
                for j in range(index_set1, MWRI_rows - 1, 1):
                    if abs(MERSI_y_vector[i] - MWRI_y_proj_vector[index_set1 + spatial_bias]) > abs(
                            MERSI_y_vector[i] - MWRI_y_proj_vector[index_set1 + 1 + spatial_bias]):
                        index_set1 += 1
                    else:
                        # print(MERSI_y_vector[i], MWRI_y_proj_vector[index_set1], MWRI_y_proj_vector[index_set1 + 1])
                        break
            y_index[i] = index_set1

        index_set1 = 0
        x_index = np.zeros(MERSI_cols).astype(np.int16)
        for i in tqdm(range(MERSI_cols)):
            if index_set1 == (MWRI_cols - 1):
                pass
            else:
                for j in range(index_set1, MWRI_cols - 1, 1):
                    if ((abs(MERSI_x_vector[i] - MWRI_x_proj_vector[index_set1 + spatial_bias])) > (
                            abs(MERSI_x_vector[i] - MWRI_x_proj_vector[index_set1 + 1 + spatial_bias]))):
                        index_set1 += 1
                    else:
                        # print(MERSI_x_vector[i], MWRI_x_proj_vector[index_set1], MWRI_x_proj_vector[index_set1 + 1])
                        break
            x_index[i] = index_set1

        MWRI_proj = MWRI_band[y_index]
        MWRI_proj = MWRI_proj[:, x_index]

        print("Generating MWRI LST preprocessed final tif...")
        gdal_writer(MERSI_data.GetProjection(), MERSI_data.GetGeoTransform(),
                    out_path, MWRI_proj, gdal.GDT_UInt16)

    @staticmethod
    def lonlat2geo(lon, lat):
        """
        将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        """

        prosrs1 = 'PROJCS["NSIDC_EASE_Grid_Global",GEOGCS["GCS_Sphere_International_1924_Authalic",' \
                  'DATUM["D_Sphere_International_1924_Authalic",SPHEROID["Sphere_International_1924_Authalic",' \
                  '6371228.0,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION[' \
                  '"Cylindrical_Equal_Area"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],' \
                  'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",30.0],UNIT["Meter",1.0]] '
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(prosrs1)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        return coords[:2]

    @staticmethod
    def point_transform_x(x):
        xy = MWRI_LST_preprocess.point_transform(3410, 32662, x, 0)
        return xy[0]

    @staticmethod
    def point_transform_y(y):
        xy = MWRI_LST_preprocess.point_transform(3410, 32662, 0, y)
        return xy[1]

    @staticmethod
    def point_transform(source_ref, target_ref, x, y):
        # 创建目标空间参考
        spatialref_target = osr.SpatialReference()
        spatialref_source = osr.SpatialReference()
        spatialref_target.ImportFromEPSG(target_ref)
        spatialref_source.ImportFromEPSG(source_ref)
        # 构建坐标转换对象，用以转换不同空间参考下的坐标
        trans = osr.CoordinateTransformation(spatialref_source, spatialref_target)
        coordinate_after_trans = trans.TransformPoint(x, y)
        return coordinate_after_trans


if __name__ == '__main__':
    test2 = MWRI_LST_preprocess(r"F:\data\FY-3D原始数据_第二批\FY3D-MWRI-LST-20230607",
                                r"F:\data\FY-3D原始数据_第二批\FY3D-DailyLst-202306-07_output\20230601.tif")
    test2.date_update()
