# parameter configure 参数页参数标识"_p"
MERSI_LST_preprocess_flag = 1
# 如已预处理MERSI数据，则置0输入预处理文件夹路径path_MERSI_preprocessed_p即可，path_MERSI_original_p可忽略；
# 否则请置1，填写未预处理图像文件夹输入路径path_MERSI_original_p，设置预处理文件夹输出路径path_MERSI_preprocessed_p。
path_MERSI_original_p = "xxxxxxx"
path_MERSI_preprocessed_p = r"xxxxxxx"  # 经过预处理的MERSI数据文件夹

MWRI_LST_preprocess_flag = 1
# 如已预处理MWRI数据，则置0输入预处理文件夹路径即可；
# 否则请置1，填写未预处理图像文件夹输入路径，设置预处理文件夹输出路径.
path_MWRI_original_p = r"xxxxxxxxx"
path_MWRI_preprocessed_p = r"xxxxxxxx"  # 经过预处理的MWRI数据文件夹

features_update_switch = 1# 特征更新开关
composition_switch = 1  # MWRI多天合成开关
injection_switch = 1  # 融合图像生成开关

update_startDate_p = "20140106"
update_endDate_p = "20141006"
update_skipDate_p = "22220101"


predict_startDate = "20140904"
predict_endDate = "20140904"

predict_composition_rule = "First" # "First" # "Mean"    "Max"  "Min"  "First" "Last"

resolution_ratio = 33
blur_windowSize_CL = 50
blur_windowSize_FL = 20   #

RC_F_window_size = 30
RC_F_shrinkage = 60

RC_C_window_size = 30
# RC_C_shrinkage = 20
RC_C_shrinkage = 1

# outputValueMin = 200
# outputValueMax  = 350

outputValueMin = -100
outputValueMax = 100

output_sdi_blur_switch = 0
include_sdi_switch = 1

MWRI_background_switch = 1  # 是否使用背景填补MWRI缺失区域，1为是，0为否。。如果为否则不用进行下方边界参数设置，例如MWRI_background。
MWRI_background = fusion_folder_p + "/background/LS_background.tif"

boundary_switch = 0  # 是否使用边界图层对区域进行掩膜，1为是，0为否。如果为否则不用进行下方边界参数设置，例如boundary_file。
boundary_country = 10  # 区域对应数字查看boundary_file文件,默认请置999
boundary_water = 255  # 水体区域需要剔除,水体区域为255
boundary_file = fusion_folder_p + "/ancillary/boundary.tif"

# Fixed parameter settings, no need to change
# HS_Slope = 0.1
# HS_intercept = 0

HS_Slope = 1
HS_intercept = 0
HS_minValue = outputValueMin
HS_maxValue = outputValueMax

# LS_Slope = 0.01
# LS_intercept = 0
# LS_minValue = 0
# LS_maxValue = 350

LS_Slope = 1
LS_intercept = 0
LS_minValue = outputValueMin
LS_maxValue = outputValueMax

update_alpha = 1.2
texture_std_limit_p = 5
sdi_std_limit_p = 10

# HS_img_nameHeadend = ""  # r"FY3D_MERSI_GBAL_L2_SST_DAY_GLL_"  # MERSI LST
# HS_img_nameBackend = ".tif"  # r"_POAD_5000M_MS.tif"
# LS_img_nameHeadend = r"FY3D_MWRIX_GBAL_L2_LST_MLT_ESD_"  # MWRI LST
# LS_img_nameBackend = r"_POAD_025KM_MS.tif.tif"
LS_img_nameHeadend = r""  # MWRI LST
LS_img_nameBackend = r".tif"
HS_img_nameHeadend = ""  # r"FY3D_MERSI_GBAL_L2_SST_DAY_GLL_"  # MERSI LST
HS_img_nameBackend = ".tif" # "_math.tif"  # r"_POAD_5000M_MS.tif"

texture_file = fusion_folder_p + "/background/texture.tif"  # Texture Feature
texture_coverage_file = fusion_folder_p + "/background/texture_coverage.tif"
texture_coverDate_file = fusion_folder_p + "/background/texture_coverDate.tif"
texture_files = [texture_file, texture_coverDate_file, texture_coverDate_file]

SDF_file = fusion_folder_p + "/background/SDI.tif"  # Sensor difference Feature
SDF_coverage_file = fusion_folder_p + r"/background/SDI_coverage.tif"
SDF_coverDate_file = fusion_folder_p + r"/background/SDI_coverDate.tif"
SDF_files = [SDF_file, SDF_coverage_file, SDF_coverDate_file]

eps = 0.0001

# texture_history_noRead = True
# SDI_history_noRead = True
