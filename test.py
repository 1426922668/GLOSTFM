# coding=utf-8
# @Time : 2023/9/7 16:50
# @Author : 陈士泽
# @File :0004_test_LST.py
# @Software : PyCharm

from config import *
from preprocess_MERSI_LST import *
from preprocess_MWRI_LST import *
from glostfm import *

if __name__ == '__main__':
    #################
    # MERSI 数据预处理
    ################
    if MERSI_LST_preprocess_flag:
        MERSI_data = MERSI_LST_preprocess(path_MERSI_original_p, path_MERSI_preprocessed_p, 20230601, 20230801,
                                          left="7", right="D", up="5", down="1")
        path_MERSI_preprocessed = MERSI_data.date_update()  # 此时预处理好的MERSI文件已经在返回的路径文件夹中
        print("MERSI 预处理过程已完成...")
    else:
        path_MERSI_preprocessed = path_MERSI_preprocessed_p
        print("MERSI 预处理过程将按设置跳过...")

    #  自动获取参考高空间分辨率图像路径 ref_HS_tif
    MERSI_preprocessed_list = os.listdir(path_MERSI_preprocessed_p)
    for i in range(len(MERSI_preprocessed_list)):
        if os.path.splitext(MERSI_preprocessed_list[i])[1].lower() == ".tif":
            ref_HS_tif = path_MERSI_preprocessed_p + r"\\" + MERSI_preprocessed_list[i]
            break

    ################
    # MWRI 数据预处理
    ################
    # MWRI与MERSI处理的先后顺序不能更换，这是由于MWRI要按照MERSI做参考生产
    if MWRI_LST_preprocess_flag:
        MWRI_data = MWRI_LST_preprocess(path_MWRI_original_p, path_MWRI_preprocessed_p, ref_HS_tif)
        path_MWRI_preprocessed = MWRI_data.date_update()  # 此时预处理好的MWRI文件已经在返回的路径文件夹中
        print("MWRI 预处理过程已完成...")
    else:
        path_MWRI_preprocessed = path_MWRI_preprocessed_p
        print("MWRI 预处理过程将按设置跳过...")

    #################
    # 时空融合模型运行
    ################

    # 融合模型路径设置
    HS_img_folder = path_MERSI_preprocessed  # 融合所需的高分辨率数据文件夹路径
    LS_img_folder = path_MWRI_preprocessed   # 融合所需的低分辨率数据文件夹路径
    fusion_folder = fusion_folder_p  # 融合数据存放文件夹，由参数（a000_Parameters）页指定
    update_startDate = update_startDate_p  # 特征更新开始时间，更新时间段应在融合日期附近
    update_endDate = update_endDate_p  # 特征更新结束时间，更新时间段应在融合日期附近
    update_skipDate = update_skipDate_p # 特征更新中需要跳过的日期，如果没有则置20002211等空日期（不存在的日期）即可

    SSTFM_test1 = UpdatesAndCompositionAndInjection(fusion_folder, HS_img_folder, LS_img_folder, ref_HS_tif)
    if features_update_switch:
        print("融合特征更新中...")
        SSTFM_test1.features_update(update_startDate, update_endDate, update_skipDate)

    if composition_switch:
        print("融合时间段图像合并...")
        LS_prediction_array = SSTFM_test1.composition(predict_startDate, predict_endDate, predict_composition_rule)

    if injection_switch:
        print("融合影像生成...")
        SSTFM_test1.injection(LS_prediction_array, blur_windowSize_CL,include_sdi=include_sdi_switch)



