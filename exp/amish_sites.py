from pathlib import  Path
import pandas as pd
import re

miami_path = Path('/opt/data/workingdir/nrakocz/oct/miami_imgs_e2e_par_full')
upen_path = Path('/opt/data/workingdir/nrakocz/oct/upen_imgs_e2e_par_full/')
ohio_path = Path('/opt/data/workingdir/nrakocz/oct/ohio_imgs_e2e_par_full_Line_pred/')
ohio_line_path = Path('/opt/data/workingdir/nrakocz/oct/ohio_imgs_e2e_par_full_Line_pred/')
ohio_cube_path = Path('/opt/data/workingdir/nrakocz/oct/ohio_imgs_e2e_par_full_Cube_pred/')


amish_sites = ['Miami','Ohio','UPENN']
amish_paths = [miami_path,ohio_path,upen_path]
amish_cube_paths = [miami_path/'Cube',ohio_cube_path,upen_path/'Cube']


# cat_cols = ['CO_SUBRETINAL_OD',
#                  'CO_SUBRETINAL_OS',
#                  'CO_SRTSRHRM_OD',
#                  'CO_SRTSRHRM_OS',
#                  'CO_INTRA_RCS_OD',
#                  'CO_INTRA_RCS_OS',
#                  'CO_OUTER_RT_OD',
#                  'CO_OUTER_RT_OS',
#                  'CO_SR_DRUSEN_OD',
#                  'CO_SR_DRUSEN_OS',
#                  'CO_HRF_IRHRFOND_OD',
#                  'CO_HRF_IRHRFOND_OS',
#                  'CO_HRF_HRFOD_OD',
#                  'CO_HRF_HRFOD_OS',
#                  'CO_PED_DPED_OD',
#                  'CO_PED_DPED_OS',
#                  'CO_PED_HPED_OD',
#                  'CO_PED_HPED_OS',
#                  'CO_PED_SEROUS_OD',
#                  'CO_PED_SEROUS_OS',
#                  'SO_SUBRETINAL_OD',
#                  'SO_SUBRETINAL_OS',
#                  'SO_SRTSRHRM_OD',
#                  'SO_SRTSRHRM_OS',
#                  'SO_INTRA_RCS_OD',
#                  'SO_INTRA_RCS_OS',
#                  'SO_OUTER_RT_OD',
#                  'SO_OUTER_RT_OS',
#                  'SO_SR_DRUSEN_OD',
#                  'SO_SR_DRUSEN_OS',
#                  'SO_HRF_IRHRFOND_OD',
#                  'SO_HRF_IRHRFOND_OS',
#                  'SO_HRF_HRFOD_OD',
#                  'SO_HRF_HRFOD_OS',
#                  'SO_PED_DPED_OD',
#                  'SO_PED_DPED_OS',
#                  'SO_PED_HPED_OD',
#                  'SO_PED_HPED_OS',
#                  'SO_PED_SEROUS_OD',
#                  'SO_PED_SEROUS_OS',
#             'CO_RPE_V3MM_L0.03_OD',
#             'CO_RPE_V3MM_L0.03_OS',
#             'CO_Drusen_Core_OD',
#             'CO_Drusen_Core_OS',
#             'IRP_RP_OD',
#             'IRP_RP_OS'
#                ]

# reg_cols = [
#     'CO_RPE_A3MM',
#     'CO_RPE_A5MM',
#     'CO_RPE_V3MM',
#     'CO_RPE_V5MM',
#     'CO_GA_A5MM',
# ]

