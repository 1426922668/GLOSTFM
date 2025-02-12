# -*- codeing = utf-8 -*-
# coding=utf-8
# @Time : 2022/5/25 15:02
# @Author : 陈士泽
# @File :normal.py
# @Software : PyCharm


import shutil
import numpy as np
import os
import random
import cv2
from osgeo import gdal


# import matplotlib
from matplotlib import cm
# matplotlib.use("agg")
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error
import datetime

# gdal库读取遥感影像数据
def image_open(img):
    data = gdal.Open(img)
    if data == 'None':
        print("图像无法读取")
    return data


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def tifFolder2array(tifFolder):
    """
    :param  tifFolder:tif所在文件夹，最终一个文件中的多个tif转为一个3维数组
    """
    tifNameList = sorted(os.listdir(tifFolder))
    flag = 0
    for i in range(len(tifNameList)):
        if os.path.splitext(tifNameList[i])[1].lower() == ".tif":
            if flag == 0:
                Indata = image_open(tifFolder + '/' + tifNameList[i])
                cols = Indata.RasterXSize
                rows = Indata.RasterYSize
                array = np.zeros([len(tifNameList), rows, cols])
                flag = 1

            path = tifFolder + "/" + tifNameList[i]
            ds = image_open(path)
            band = ds.GetRasterBand(1)
            data_band = band.ReadAsArray(0, 0, cols, rows)
            array[i, :, :] = data_band
    print(tifFolder + '数据读取完毕!')
    return array


def array2tifFolder(array_3d, ref_folder_path, output_folder_path=None):
    """
    :param  out_folder:3维数组转为该文件夹中多个tif，3维数组1d为tif序号，2d为tif的col号？，3d为tif的row号？
    :param  ref_folder:参考文件夹，提供tif文件对应名称、坐标参考，实际上，输出的为文件夹
    """
    if not output_folder_path:
        output_folder_path = ref_folder_path + "_proc"
    mkdir(output_folder_path)
    tifNameList = os.listdir(ref_folder_path)
    ref = sorted(os.listdir(ref_folder_path))
    ref_data = image_open(ref_folder_path + '/' + ref[0])
    for i in range(len(tifNameList)):
        #  判断当前文件是否为tif文件
        if os.path.splitext(tifNameList[i])[1].lower() == ".tif":
            output_tif_path = output_folder_path + "/" + tifNameList[i]
            gdal_writer(ref_data.GetProjection(), ref_data.GetGeoTransform(),
                        output_tif_path, array_3d[i], gdal.GDT_Float32)
    print(output_folder_path + '数据写入完毕!')


def rndm_smpl_1d(array1, array2, smpl_num):
    array1 = array1.flatten()
    array2 = array2.flatten()
    indices_sample = [0] * (len(array1) - smpl_num) + [1] * smpl_num
    random.shuffle(indices_sample)
    smplarray1 = np.extract(indices_sample, array1)
    smplarray2 = np.extract(indices_sample, array2)
    return smplarray1, smplarray2


def gdal_writer(Proj, GeoTrans, path, array, data_type=gdal.GDT_Float32):
    gtiff_driver = gdal.GetDriverByName("GTiff")
    out_ds = gtiff_driver.Create(path, array.shape[1], array.shape[0], 1, data_type)
    # 设置输出数据坐标投影为原始影像的坐标投影
    out_ds.SetProjection(Proj)
    out_ds.SetGeoTransform(GeoTrans)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_ds.FlushCache()


def warp_folder(shp_path, input_folder_path, output_folder_path=None):
    # 用于批量掩膜的函数
    """
    :param input_folder_path:待裁剪tif集文件夹
    :param output_folder_path:裁剪输出文件夹路径
    :param shp_path:  用于裁剪的shp文件路径
    """
    if not output_folder_path:
        output_folder_path = input_folder_path + "_warp"
    mkdir(output_folder_path)
    tifNameList = os.listdir(input_folder_path)
    for i in range(len(tifNameList)):
        #  判断当前文件是否为tif文件
        if os.path.splitext(tifNameList[i])[1].lower() == ".tif":
            input_tif_path = input_folder_path + "/" + tifNameList[i]
            output_tif_path = output_folder_path + "/" + tifNameList[i]
            ds = gdal.Warp(output_tif_path,
                           input_tif_path,
                           warpMemoryLimit=500, #内存大小M
                           format='GTiff',  # 保存图像的格式
                           cutlineDSName=shp_path,  # 矢量文件的完整路径
                           cropToCutline=True,
                           copyMetadata=True,
                           creationOptions=['COMPRESS=LZW', "TILED=True"],
                           dstNodata=0)
    print("Warp done!")

def shp_clip_tif(inMaskData, in_folder,out_folder):
    ## note end need '\\'
    # from arcpy.sa import *
    # import os
    in_folder = in_folder + r"\\"
    out_folder = out_folder + r"\\"  #???

    tifNameList = sorted(os.listdir(in_folder))
    for i in range(len(tifNameList)):
        in_file = in_folder + tifNameList[i]
        out_file = out_folder + tifNameList[i][17:25]+".tif"
        outExtractByMask = ExtractByMask(in_file, inMaskData)
        outExtractByMask.save(out_file)

def tif_min_max(tif_path):
    ds = image_open(tif_path)
    band = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    print(np.min(band), np.max(band))

def tif_show(tif_path):
    ds = image_open(tif_path)
    band = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    plt.imshow(band)#,vmin =-10 ,vmax=70)#,cmap="gist_ncar")#cmap="nipy_spectral")#,cmap="viridis_r")
    plt.show()

def tif2png(tif_path, dpi_png=900):
    ds = image_open(tif_path)
    band = ds.GetRasterBand(1).ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.imshow(band,cmap="viridis")#vmin =-15,vmax=65,)
    plt.savefig((tif_path.split("/")[-1]).split(".")[-2] + ".png", dpi=dpi_png, bbox_inches='tight')

def tifs2pngs(folder_path):
    tifs = sorted(os.listdir(folder_path))
    for i in range(len(tifs)):
        tif2png(folder_path + "/" + tifs[i])

def run_imap_mp(func, argument_list, num_processes='', is_tqdm=True):
    '''
    多进程与进度条结合

    param:
    ------
    func:function
        函数
    argument_list:list
        参数列表
    num_processes:int
        进程数，不填默认为总核心-3
    is_tqdm:bool
        是否展示进度条，默认展示
    '''
    result_list_tqdm = []
    try:
        import multiprocessing
        if num_processes == '':
            num_processes = multiprocessing.cpu_count() - 3
        pool = multiprocessing.Pool(processes=num_processes)
        if is_tqdm:
            from tqdm import tqdm
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
        else:
            for result in pool.imap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
        pool.join()
    except:
        result_list_tqdm = list(map(func, argument_list))
        print("failed")
    return result_list_tqdm


def evaluation(origin_array, fill_array, noValue=0):
    # origin_array, fill_array = delete_miss(origin_array, fill_array, noValue)
    rmse_const = mean_squared_error(origin_array, fill_array) ** 0.5
    ssim_const = ssim(origin_array, fill_array)
    print("rmse_const:", rmse_const)
    print("ssim_const:", ssim_const)
    return rmse_const, ssim_const


def bandmath(tifFolder, tifType, outputFolder = None):
    """"
    :param outputFolder:
    :param  tifFolder:为待波段计算影像集文件夹地址
    :param tifType: 对应实验数据类型，应输入“MODIS”或“L8”以进行相应运算
    """
    if outputFolder == None:
        mathOutputFolder = tifFolder + "_math"
    else:
        mathOutputFolder = outputFolder

    mkdir(mathOutputFolder)
    tifNameList = os.listdir(tifFolder)
    for i in range(len(tifNameList)):
        #  判断当前文件是否为tif文件
        if os.path.splitext(tifNameList[i])[1].lower() == ".tif":
            tifPath = tifFolder + "/" + tifNameList[i]
            in_ds = gdal.Open(tifPath)
            inBand = in_ds.GetRasterBand(1)
            inImg = inBand.ReadAsArray().astype(np.float32)

            if tifType == "MODIS":
                outValue = 0.02 * inImg - 273.15
            elif tifType == "L8":
                outValue = 0.00341802 * inImg + 149 - 273.15
            else:
                print("请输入指定类型")
                break
            gtiff_driver = gdal.GetDriverByName("GTiff")
            out_ds = gtiff_driver.Create(mathOutputFolder + "/" + os.path.splitext(tifNameList[i])[0] + ".tif",
                                         outValue.shape[1], outValue.shape[0], 1, gdal.GDT_Float32)
            # 设置输出数据坐标投影为原始影像的坐标投影
            out_ds.SetProjection(in_ds.GetProjection())
            out_ds.SetGeoTransform(in_ds.GetGeoTransform())
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outValue)
            out_band.FlushCache()
            print("done!")
    return mathOutputFolder


def delete_miss(a, b, value=0):
    a, b = a.flatten(), b.flatten()
    a1, b1 = np.array([]), np.array([])
    for i in range(len(a)):
        if a[i] > value and b[i] > value:
            a1 = np.append(a1, a[i])
            b1 = np.append(b1, b[i])
    return a1, b1


def ols(x_array, y_array):
    e = np.ones(len(x_array))
    A = np.hstack((x_array.reshape(-1, 1), e.reshape(-1, 1)))
    Y = np.reshape(y_array, (-1, 1))
    # ## 2、求解系数
    w = np.matmul(np.transpose(A), A)  # w = A^T*A
    w1 = np.linalg.inv(w)  # 矩阵求逆
    w2 = np.matmul(w1, np.transpose(A))
    solution = np.matmul(w2, Y)
    return solution

def delete_noValue_files(Folder_Path,novalue = -200):
    file = sorted(os.listdir(Folder_Path))
    file_all = tifFolder2array(Folder_Path)

    #删除有缺失值的影像文件
    for i in range(len(file)):
        if min(file_all[i].flatten()) < novalue:
            os.remove(Folder_Path+"/"+file[i])
        else:
            os.rename(Folder_Path + "/" + file[i], Folder_Path + "/" + str(file[i])[9:16] + ".tif")

    print("delete_noValue_files done!")

def delete_lowPixelNum_files(Folder_Path,PixelNum=int(36*36*0.5),novalue = -200):# 删除低于指定有效像元数的图像
    file = sorted(os.listdir(Folder_Path))
    file_all = tifFolder2array(Folder_Path)

    #删除缺失值过多的影像文件
    for i in range(len(file)):
        if np.sum(np.where(file_all[i].flatten() > novalue,1,0)) < PixelNum:
            os.remove(Folder_Path+"/"+file[i])
        else:
            os.rename(Folder_Path + "/" + file[i], Folder_Path + "/" + str(file[i])[9:16] + ".tif")

    print("delete_lowPixelNum_files done!")

def pixelPCT(Folder_Path, outputFolder, novalue = -200):# 统计各图像有效像元占比
    file = sorted(os.listdir(Folder_Path))
    file_all = tifFolder2array(Folder_Path)
    pixelPCT_array = np.zeros(len(file))
    for i in range(len(file)):
        pixelPCT_array[i] = np.sum(np.where(file_all[i].flatten() > novalue,1,0))/len(file_all[i].flatten())
    np.save(outputFolder + r"\pixelPCT.npy",pixelPCT_array)
    np.savetxt(outputFolder + "\pixelPCT.txt", pixelPCT_array)
    print("done!")

def delete_lowPixelPCT_files(Folder_Path, pixelPCT_Path, valuePCT=0.9):# 删除低于指定有效像元数的图像
    file = sorted(os.listdir(Folder_Path))
    file_all = tifFolder2array(Folder_Path)
    arrayPCT = np.load(pixelPCT_Path)
    #删除缺失值过多的影像文件
    for i in range(len(file)):
        if arrayPCT[i] < valuePCT:
            os.remove(Folder_Path+"/"+file[i])
        else:
            pass
    print("done!")

def statPixelVary(Folder_Path,x =600,y=600 ):
    print("go")
    file_all = tifFolder2array(Folder_Path)
    arrayStat = np.zeros(len(file_all))
    for i in range(len(file_all)):
        arrayStat[i] = file_all[i, x, y]
    np.savetxt("stat.txt", arrayStat)
    print("done")

# statPixelVary(r"E:\data\MOD11A1\BJ_2013-2020all_Filled_0.9_predict")

def diffImg(reaImgPath,preImgPath):
    outputPath= preImgPath[:-4]+"diff.tif"
    realImgData = image_open(reaImgPath)
    arrayRealImg= realImgData.GetRasterBand(1).ReadAsArray(0, 0, realImgData.RasterXSize, realImgData.RasterYSize)
    preImgData = image_open(preImgPath)
    arrayPreImg= preImgData.GetRasterBand(1).ReadAsArray(0, 0, preImgData.RasterXSize, preImgData.RasterYSize)
    gdal_writer(realImgData.GetProjection(), realImgData.GetGeoTransform(), outputPath,arrayPreImg - arrayRealImg, gdal.GDT_Float32)
    print("done!")


def setImgProj(refImgPath,projImgPath):
    outputPath= projImgPath[:-4]+"proj.tif"
    refImgData = image_open(refImgPath)
    projImgData = image_open(projImgPath)
    arrayProjImg= projImgData.GetRasterBand(1).ReadAsArray(0, 0, projImgData.RasterXSize, projImgData.RasterYSize)
    gdal_writer(refImgData.GetProjection(), refImgData.GetGeoTransform(), outputPath,arrayProjImg, gdal.GDT_Float32)
    print("done!")

#projImg(r"E:\data\Landsat\LC08_L2SP_123039_20220505_20220511_02_T1_ST_B10.TIF",r"E:\data\Landsat\LC09_L2SP_122039_20211112_20220119_02_T1_ST_B10.TIF")

def reNameAndDelTifs(MODs_Path,Lsts_Path ):
    # 初始化
    MOD = sorted(os.listdir(MODs_Path))
    Lst = sorted(os.listdir(Lsts_Path))

    # 重命名MOD影像文件
    for i in range(len(MOD)):
            os.rename(MODs_Path + "/" + MOD[i], MODs_Path + "/" + str(MOD[i])[9:16] + "_MODIS.tif")

    for i in range(len(Lst)):
            os.rename(Lsts_Path + "/" + Lst[i], Lsts_Path + "/" + str(Lst[i])[17:25] + "_Landsat.tif")

    # 将Landsat文件与MOD文件对应，多余Landsat文件删除
    num = 0
    for i in range((len(Lst))):
        flag = 0
        for p in range(len(MOD)):
            if int(str(MOD[p])[4:7]) == int(
                    datetime.date(int(str(Lst[i])[0:4]), int(str(Lst[i])[4:6]), int(str(Lst[i])[6:8])).strftime(
                            "%j")) and int(str(MOD[p])[0:4]) == int(str(Lst[i])[0:4]):
                # print(int(str(MOD[p])[4:7]),
                #       int(datetime.date(int(str(Lst[i])[0:4]), int(str(Lst[i])[4:6]), int(str(Lst[i])[6:8])).strftime(
                #           "%j")))
                flag = 1
        if flag == 0:
            os.remove(Lsts_Path + "/" + Lst[i])
            num += 1
    print("Done!" )

def reNameAndDelTifs_v2(MODs_Path,Lsts_Path ):
    # 初始化
    MOD = sorted(os.listdir(MODs_Path))
    Lst = sorted(os.listdir(Lsts_Path))

    # 重命名MOD影像文件
    for i in range(len(MOD)):
            os.rename(MODs_Path + "/" + MOD[i], MODs_Path + "/" + str(MOD[i])[9:16] + "_MODIS.tif")

    for i in range(len(Lst)):
            os.rename(Lsts_Path + "/" + Lst[i], Lsts_Path + "/" + str(Lst[i])[17:25] + "_Landsat.tif")

    # 将Landsat文件与MOD文件对应，多余Landsat文件删除
    num = 0
    for i in range((len(Lst))):
        flag = 0
        for p in range(len(MOD)):
            if int(str(MOD[p])[4:7]) == int(
                    datetime.date(int(str(Lst[i])[0:4]), int(str(Lst[i])[4:6]), int(str(Lst[i])[6:8])).strftime(
                            "%j")) and int(str(MOD[p])[0:4]) == int(str(Lst[i])[0:4]):
                # print(int(str(MOD[p])[4:7]),
                #       int(datetime.date(int(str(Lst[i])[0:4]), int(str(Lst[i])[4:6]), int(str(Lst[i])[6:8])).strftime(
                #           "%j")))
                flag = 1
        if flag == 0:
            os.remove(Lsts_Path + "/" + Lst[i])
            num += 1
    print("Done!" )

def reSample0(resampleFolder,refTifPath):#lotsfm没有强制要求将低分辨率采样至与高分辨一致，但其他模型有要求因此写此函数以满足其他模型
    rData = sorted(os.listdir(resampleFolder))
    rData0 = image_open(resampleFolder + '/' + rData[0])
    fData0 = image_open(refTifPath)
    rData_cols, rData_rows = rData0.RasterXSize, rData0.RasterYSize
    fData_cols, fData_rows = fData0.RasterXSize, fData0.RasterYSize
    rData_all = tifFolder2array(resampleFolder)

    rData_NEAREST= np.zeros([len(rData_all), fData_cols, fData_rows])
    for i in range(len(rData)):
        rData_NEAREST[i] = cv2.resize(rData_all[i], [fData_cols, fData_rows], interpolation=cv2.INTER_NEAREST)
        gdal_writer(fData0.GetProjection(), fData0.GetGeoTransform(), resampleFolder + '/' + str(rData[i])[:-4]+"_resample.tif",rData_NEAREST[i], gdal.GDT_Float32)
    print("Done!")

def dataNumAndDrift(MODFolder,driftnpyPath):
    MOD = sorted(os.listdir(MODFolder))
    dateNum=[]
    for i in range(len(MOD)):
        dateNum.append(str(MOD[i])[4:7])
    np.savetxt("dateNum.txt", dateNum,fmt = "%s")
    a = np.load(driftnpyPath)
    c =np.sum(np.sum(a,1),1)/36/36
    np.savetxt("drift_sum.txt", c)

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

def num2str(num_int):
    if num_int < 10:
        num_str = "0" + str(num_int)
    else:
        num_str = str(num_int)
    return num_str

def del_folder(dir_path):
    shutil.rmtree(dir_path)

def xxxx_xx_xx2xxxx_xxx(year,day):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    import datetime
    import datetime as dt
    first_day=datetime.datetime(year,1,1)
    add_day=datetime.timedelta(days=day-1)
    return str(datetime.datetime.strftime(first_day+add_day,"%Y%m%d"))

def yyyy_mm_dd2yyyy_ddd(yyyy_mm_dd):
    yyyy_mm_dd = str(yyyy_mm_dd)
    yyyy = int(yyyy_mm_dd[:4])
    mm = int(yyyy_mm_dd[4:6])
    dd = int(yyyy_mm_dd[6:8])
    yyyy_mm_dd_datetime = datetime.datetime(yyyy, mm, dd)
    start_datetime = datetime.datetime(yyyy,1,1)
    ddd = (yyyy_mm_dd_datetime-start_datetime).days +1
    if ddd<10:
        ddd = "00" + str(ddd)
    elif ddd<100:
        ddd = "0" + str(ddd)
    else:
        ddd = str(ddd)
    yyyy_ddd = str(yyyy) + ddd
    return yyyy_ddd

def yyyy_ddd2yyy_mm_dd(yyyy_ddd):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    year = int(yyyy_ddd[:4])
    day = int(yyyy_ddd[4:7])
    first_day = datetime.datetime(year, 1, 1)
    add_day = datetime.timedelta(days=day - 1)
    yyyy_mm_dd = str(datetime.datetime.strftime(first_day + add_day, "%Y%m%d"))
    return yyyy_mm_dd

def missFill(MODsPath, FillsPath,noValue = -200, window_size = 36):

    start_time = datetime.datetime.now()

    MOD = sorted(os.listdir(MODsPath))
    MODIndata = image_open(MODsPath + '/' + MOD[0])
    cols = MODIndata.RasterXSize
    rows = MODIndata.RasterYSize
    mData = tifFolder2array(MODsPath)
    fillData = mData.copy()
    fill_num = 0
    for x in range(cols):
        for y in range(rows):
            numValue = np.sum(np.where(mData[:, x, y] < noValue, 0, 1))
            if numValue < 2 or numValue == len(mData[:, 0, 0]):
                continue
            pointsCor = np.zeros(window_size**2)
            for m in range(window_size):
                m1 = x - window_size // 2 + m
                if m1 >= cols or m1 < 0:
                    continue
                for n in range(window_size):
                    n1 = y - window_size // 2 + n
                    if n1 >= rows or n1 < 0:
                        continue
                    xArray, yArray = delete_miss(mData[:, m1, n1], mData[:, x, y],noValue)
                    if len(yArray) < 3 or np.std(xArray) == 0 or np.std(yArray) == 0:
                        pass
                    else:
                        pointsCor[m*window_size+n] = np.corrcoef(xArray, yArray)[0, 1]#!!!!!!!!
            sorted_id = sorted(range(len(pointsCor)), key=lambda x: pointsCor[x], reverse=True)

            for j in range(len(pointsCor)):
                m = sorted_id[j] // window_size
                n = sorted_id[j] % window_size
                m1 = x - window_size // 2 + m
                n1 = y - window_size // 2 + n
                if m1 >= cols or n1 >= rows or m1 < 0 or n1 < 0:
                    continue
                if np.sum(np.where(fillData[:, x, y]<noValue, 0, 1)-np.where(fillData[:, m1, n1] < noValue, 0, 1)) != 0:
                    xArray, yArray = delete_miss(fillData[:, m1, n1], fillData[:, x, y], noValue)
                    solution = ols(xArray, yArray)
                    for i in range(len(mData)):
                        if fillData[i, x, y] <= noValue < fillData[i, m1, n1]:
                            fillData[i, x, y] = solution[0][0] * fillData[i, m1, n1] + solution[1][0]
                            fill_num = fill_num + 1
                            # print(fill_num)
                elif np.sum(np.where(fillData[:, x, y]<noValue, 1, 0)) == 0:
                    break
                else:
                    continue

        for x in range(cols-1, -1, -1):
            for y in range(rows-1, -1, -1):
                numValue = np.sum(np.where(mData[:, x, y] < noValue, 0, 1))
                if numValue < 2 or numValue == len(mData[:, 0, 0]):
                    continue
                pointsCor = np.zeros(window_size ** 2)
                for m in range(window_size):
                    m1 = x - window_size // 2 + m
                    if m1 >= cols or m1 < 0:
                        continue
                    for n in range(window_size):
                        n1 = y - window_size // 2 + n
                        if n1 >= rows or n1 < 0:
                            continue
                        xArray, yArray = delete_miss(mData[:, m1, n1], mData[:, x, y], noValue)
                        if len(yArray) < 3 or np.std(xArray) == 0 or np.std(yArray) == 0:
                            pass
                        else:
                            pointsCor[m * window_size + n] = np.corrcoef(xArray, yArray)[0, 1]  # !!!!!!!!
                sorted_id = sorted(range(len(pointsCor)), key=lambda x: pointsCor[x], reverse=True)

                for j in range(len(pointsCor)):
                    m = sorted_id[j] // window_size
                    n = sorted_id[j] % window_size
                    m1 = x - window_size // 2 + m
                    n1 = y - window_size // 2 + n
                    if m1 >= cols or n1 >= rows or m1 < 0 or n1 < 0:
                        continue
                    if np.sum(np.where(fillData[:, x, y] < noValue, 0, 1) - np.where(fillData[:, m1, n1] < noValue, 0,
                                                                                     1)) != 0:
                        xArray, yArray = delete_miss(fillData[:, m1, n1], fillData[:, x, y], noValue)
                        solution = ols(xArray, yArray)
                        for i in range(len(mData)):
                            if fillData[i, x, y] <= noValue < fillData[i, m1, n1]:
                                fillData[i, x, y] = solution[0][0] * fillData[i, m1, n1] + solution[1][0]
                                fill_num = fill_num + 1
                                #print(fill_num)
                    elif np.sum(np.where(fillData[:, x, y] < noValue, 1, 0)) == 0:
                        break
                    else:
                        continue

    for i in range(len(MOD)):
        gdal_writer(MODIndata.GetProjection(), MODIndata.GetGeoTransform(), FillsPath + "/" + MOD[i], fillData[i])
    print("fill done!")
    last_time = datetime.datetime.now()
    print(last_time - start_time)

def out_date_by_day(year, day):
    '''
    根据输入的年份和天数计算对应的日期
    '''
    first_day = datetime.datetime(year, 1, 1)
    add_day = datetime.timedelta(days=day - 1)
    return str(datetime.datetime.strftime(first_day + add_day, "%Y%m%d"))

def renameMODIS(in_file, yyyyPositionStart = 0, yyyyPositionEnd = 3, dddPositionStart = 4, dddPositionEnd = 6):
    tifNameList = sorted(os.listdir(in_file))
    for i in range(len(tifNameList)):
        print(tifNameList[i])
        filename = tifNameList[i]
        # newname = "FY3D_MWRIX_GBAL_L2_LST_MLT_ESD_" + out_date_by_day(int(tifNameList[i][:4]),int(tifNameList[i][4:7]))+"_POAD_025KM_MS.tif"
        # newname = out_date_by_day(int(tifNameList[i][:4]), int(tifNameList[i][4:7])) + ".tif"
        # newname = filename[:8] + ".tif"
        newname = out_date_by_day(int(tifNameList[i][yyyyPositionStart:yyyyPositionEnd+1]),
                                  int(tifNameList[i][dddPositionStart:dddPositionEnd+1])) + ".tif"

        print(newname)
        os.rename(in_file + "\\" + tifNameList[i], in_file + "\\" + newname)

if  __name__ == "__main__":
    # reSample0(r"C:\Users\chen1\Desktop\wuhantest",r"C:\Users\chen1\Desktop\20171030_Landsat.tif")
    # diffImg(r"C:\Users\chen1\Desktop\wuhantest\20191020_Landsat.tif"
    #         , r"C:\Users\chen1\Desktop\wuhantest\2019293_predict.tif")
    # reNameAndDelTifs(r"E:\data\LOTSFM\train_SH_error\M_train",r"E:\data\Landsat\WuHan_del_warp_math_rename")

    # projImg(r"D:\work\data\research\03_夏季极端高温中北京地物地表温度表现\时空融合栅格图\tifs_预测与真实的差值图\2022\2022029_0math.tif",r"D:\work\data\research\03_夏季极端高温中北京地物地表温度表现\时空融合栅格图\tifs_各算法融合结果图像\2018098_ganstfm1.tif")


    # pixelPCT("E:\data\MOD11A1\WH_MRT_warp_math - 副本")
    # delete_lowPixelPCT_files("E:\data\MOD11A1\WH_MRT_warp_math_PCT","E:\csz\code\LastLOTSFM_2\pixelPCT_WH.npy")

    # warp_folder(r"E:\data\Shp\WuHan36x36\WuHan36x36.shp",r"E:\data\Landsat\新建文件夹")
    # bandmath(r"E:\data\Landsat\WuHan_del_warp","L8")
    # delete_noValue_files(r"E:\data\MOD11A1\BJ_2013-2020all_Filled_del")
    # delete_lowPixelNum_files(r"C:\Users\chen1\Desktop\BJ_2013-2020all_MRT_warp_math")
    # pixelPCT(r"C:\Users\chen1\Desktop\BJ_2013-2020all_MRT_warp_math")
    # delete_lowPixelPCT_files(r"C:\Users\chen1\Desktop\BJ_2013-2020all_Filled","pixelPCT.npy")
    pass