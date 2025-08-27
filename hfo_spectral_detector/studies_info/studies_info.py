
import os
import socket
import pandas as pd
import numpy as np
from pathlib import Path

class StudiesInfo:
    def __init__(self) -> None:
        pass
    
    def Maggi_Stroke_Data_1(self):
        dataset_name = "Maggi_Stroke_Data_1"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_1/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_1/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BAI~ Male_18852025-4e7c-4771-b29c-1680b3c248b6_3.2.edf", "BAI~ Male_18852025-4e7c-4771-b29c-1680b3c248b6_3.2_annotations.mat"],
        ["BAI~ Male_488594c9-ce54-44fe-87cd-e7e737ac61a1_clip2corr.edf", "BAI~ Male_488594c9-ce54-44fe-87cd-e7e737ac61a1_clip2corr_annotations.mat"],
        ["BAI~ Male_58bc137c-0b47-4d92-bbd7-62755aba5b8b_clip1.1.edf", "BAI~ Male_58bc137c-0b47-4d92-bbd7-62755aba5b8b_clip1.1_annotations.mat"],
        ["BAI~ Male_d65a5a12-3139-4d9f-ba62-85692393b768_clip3.1.edf", "BAI~ Male_d65a5a12-3139-4d9f-ba62-85692393b768_clip3.1_annotations.mat"],
        ["DOLAN~ Matteo_338baf73-b835-4d5a-844d-230fd11b6f9d_clip2longcheckSNR.edf", "DOLAN~ Matteo_338baf73-b835-4d5a-844d-230fd11b6f9d_clip2longcheckSNR_annotations.mat"],
        ["DOLAN~ Matteo_ac6ef925-7cdd-45e7-9df0-4ec1f61d5dfa_clip3.edf", "DOLAN~ Matteo_ac6ef925-7cdd-45e7-9df0-4ec1f61d5dfa_clip3_annotations.mat"],
        ["FLORESLEDO~ Fe_4f59ff22-bf9a-42bd-afe2-81a75afbbce1_clip3.2.edf", "FLORESLEDO~ Fe_4f59ff22-bf9a-42bd-afe2-81a75afbbce1_clip3.2_annotations.mat"],
        ["FLORESLEDO~ Fe_77328de8-32a8-40ba-befd-cc28e21e169d_clip2.edf", "FLORESLEDO~ Fe_77328de8-32a8-40ba-befd-cc28e21e169d_clip2_annotations.mat"],
        ["FLORESLEDO~ Fe_fd4fbe0a-d55a-4607-be05-0977ed1f09de_clip1extra short.edf", "FLORESLEDO~ Fe_fd4fbe0a-d55a-4607-be05-0977ed1f09de_clip1extra short_annotations.mat"],
        ["GERMERSHAUSEN~_b9c91e38-8fe4-4f8d-ad84-7be9f91d2773_clip1.edf", "GERMERSHAUSEN~_b9c91e38-8fe4-4f8d-ad84-7be9f91d2773_clip1_annotations.mat"],
        ["KAUR~ Male_425002f3-2232-440f-8a2f-900fadf664c3_clip2.edf", "KAUR~ Male_425002f3-2232-440f-8a2f-900fadf664c3_clip2_annotations.mat"],
        ["KAUR~ Male_bb3050be-5569-48ef-853d-858dc766e0a5_clip3.edf", "KAUR~ Male_bb3050be-5569-48ef-853d-858dc766e0a5_clip3_annotations.mat"],
        ["KAUR~ Male_ef26672f-9961-4b85-b961-2d125f70f69c_clip1.edf", "KAUR~ Male_ef26672f-9961-4b85-b961-2d125f70f69c_clip1_annotations.mat"],
        ["MAGNUSON~ Fema_8ba6e632-3588-4051-a449-98bd621015bb_clip1.edf", "MAGNUSON~ Fema_8ba6e632-3588-4051-a449-98bd621015bb_clip1_annotations.mat"],
        ["MAGNUSON~ Fema_c0be2f31-7282-49ba-b19d-da14e250f064_clip2.edf", "MAGNUSON~ Fema_c0be2f31-7282-49ba-b19d-da14e250f064_clip2_annotations.mat"],
        ["MCLEAN~ Male_319eb166-fdef-4b49-8738-0ec01bb259f4_clip1.edf", "MCLEAN~ Male_319eb166-fdef-4b49-8738-0ec01bb259f4_clip1_annotations.mat"],
        ["MCLEAN~ Male_3f1d31f8-c2d4-41fc-8fdb-134c22ca2838_clip3alternative.edf", "MCLEAN~ Male_3f1d31f8-c2d4-41fc-8fdb-134c22ca2838_clip3alternative_annotations.mat"],
        ["MCLEAN~ Male_559c2d9b-0af5-4dcb-8e3c-2c9515ac0b06_clip2.edf", "MCLEAN~ Male_559c2d9b-0af5-4dcb-8e3c-2c9515ac0b06_clip2_annotations.mat"],
        ["MCLEAN~ Male_de41c5b4-4c44-4872-877f-5b7b73a2e2ab_clip3.edf", "MCLEAN~ Male_de41c5b4-4c44-4872-877f-5b7b73a2e2ab_clip3_annotations.mat"],
        ["McNAUGHTON~ Ma_60fd4778-8fca-4e65-89c1-3599af0e0782_clip1.edf", "McNAUGHTON~ Ma_60fd4778-8fca-4e65-89c1-3599af0e0782_clip1_annotations.mat"],
        ["McNAUGHTON~ Ma_643439e4-8bfe-41fd-800f-63cc792679d0_clip4.edf", "McNAUGHTON~ Ma_643439e4-8bfe-41fd-800f-63cc792679d0_clip4_annotations.mat"],
        ["McNAUGHTON~ Ma_e631d7db-ca9c-4a09-b6bc-cdaa4e96960e_clip3.edf", "McNAUGHTON~ Ma_e631d7db-ca9c-4a09-b6bc-cdaa4e96960e_clip3_annotations.mat"],
        ["MEGA~ Male_635ef0ab-5eb3-42cc-a2a9-866e6fca5f65_clip1new.edf", "MEGA~ Male_635ef0ab-5eb3-42cc-a2a9-866e6fca5f65_clip1new_annotations.mat"],
        ["MEGA~ Male_73aff7f3-ed20-42c7-9f52-5b4b1c2ae931_clip3.edf", "MEGA~ Male_73aff7f3-ed20-42c7-9f52-5b4b1c2ae931_clip3_annotations_corrected.mat"],
        ["MEGA~ Male_a3bbfc58-76bf-41c9-af8c-4061a4cd206d_clip2.edf", "MEGA~ Male_a3bbfc58-76bf-41c9-af8c-4061a4cd206d_clip2_annotations.mat"],
        ["MEGA~ Male_dd9c7e99-9feb-451a-ae62-acc110ba5153_clip4.edf", "MEGA~ Male_dd9c7e99-9feb-451a-ae62-acc110ba5153_clip4_annotations.mat"],
        ["NEILSONTRUMBLE_222d821c-0540-4892-8d72-34894bf93754_clip1.edf", "NEILSONTRUMBLE_222d821c-0540-4892-8d72-34894bf93754_clip1_annotations.mat"],
        ["NEILSONTRUMBLE_2dc3d459-cea6-409e-8f73-ee2704f9cdfd_clip2.edf", "NEILSONTRUMBLE_2dc3d459-cea6-409e-8f73-ee2704f9cdfd_clip2_annotations.mat"],
        ["PRICE~ Weston_01073139-8028-4dd3-a867-199467e470d4_clip1corr1.edf", "PRICE~ Weston_01073139-8028-4dd3-a867-199467e470d4_clip1corr1_annotations.mat"],
        ["QADIR~ Fathiya_25854595-03b8-42d4-b1d3-0c1991d03911_clip5.edf", "QADIR~ Fathiya_25854595-03b8-42d4-b1d3-0c1991d03911_clip5_2annotations.mat"],
        ["QADIR~ Fathiya_307b2e7f-7fd0-4c74-b4a1-97721abcfd79_clip1.2.edf", "QADIR~ Fathiya_307b2e7f-7fd0-4c74-b4a1-97721abcfd79_clip1.2_annotations.mat"],
        ["QADIR~ Fathiya_60fe42da-518b-46e8-8cd3-a70b499f6a23_clip4.edf", "QADIR~ Fathiya_60fe42da-518b-46e8-8cd3-a70b499f6a23_clip4_annotations.mat"],
        ["QADIR~ Fathiya_a1f023d8-ce00-4d4e-a379-8f98baa145f3_clip1.edf", "QADIR~ Fathiya_a1f023d8-ce00-4d4e-a379-8f98baa145f3_clip1_annotations.mat"],
        ["QADIR~ Fathiya_b3c6d9ca-c1c2-46f9-ae42-4b7e68ac0568_clip4corrnew.edf", "QADIR~ Fathiya_b3c6d9ca-c1c2-46f9-ae42-4b7e68ac0568_clip4corrnew_annotations.mat"],
        ["QADIR~ Fathiya_c7d184ce-56a5-42c4-8c7c-a26186d60e42_clip2(long,better).edf", "QADIR~ Fathiya_c7d184ce-56a5-42c4-8c7c-a26186d60e42_clip2(long,better)_annotations_corrected.mat"],
        ["QADIR~ Fathiya_cfea5f8e-eb97-471e-a1d7-790f14684489_clip6.edf", "QADIR~ Fathiya_cfea5f8e-eb97-471e-a1d7-790f14684489_clip6_annotations.mat"],
        ["QADIR~ Fathiya_dfd6d595-3a4c-4f8c-b4af-0d196b1eb981_clip3.1.edf", "QADIR~ Fathiya_dfd6d595-3a4c-4f8c-b4af-0d196b1eb981_clip3.1_annotations.mat"],
        ["SPROTT~ Male_97836b6d-ccf0-4f8d-a22e-670701771796_clip1.edf", "SPROTT~ Male_97836b6d-ccf0-4f8d-a22e-670701771796_clip1_annotations.mat"],
        ["SPROTT~ Male_f7bc4d42-3804-4cdd-94ae-a126c2fa02b8_clip2.edf", "SPROTT~ Male_f7bc4d42-3804-4cdd-94ae-a126c2fa02b8_clip2_annotations.mat"],


        ]

        return dataset_name, data_path, eeg_files_info
    
    def Maggi_Stroke_Data_2(self):
        dataset_name = "Maggi_Stroke_Data_2"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_2_Jan2024_ForRome/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_2_Jan2024_ForRome/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["AMBROSIO~ Kyle_68375d1d-96c5-43cd-a5fa-c38547650e78.edf", "AMBROSIO~ Kyle_68375d1d-96c5-43cd-a5fa-c38547650e78_annotations.mat"],
        ["BAHTA~ Dawit_2eea09fa-d77d-4acc-8012-4aae693d48ad.edf", "BAHTA~ Dawit_2eea09fa-d77d-4acc-8012-4aae693d48ad_annotations.mat"],
        #["BHATTI~ Arjun_014d5a1a-0532-44f1-90a5-b05dce073131.edf", "BHATTI~ Arjun_014d5a1a-0532-44f1-90a5-b05dce073131_annotations.mat"], # Too Noisy
        ["DOYLE~ Hannah_efbb65d1-0ac3-4e24-bfdd-6e9f35c6dc3e.edf", "DOYLE~ Hannah_efbb65d1-0ac3-4e24-bfdd-6e9f35c6dc3e_annotations.mat"],
        ["ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6.edf", "ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotations.mat"],
        ["GAREAU~ Braede_1007dcde-cd78-447d-99f7-823d82eec545_LTM1norm.edf", "GAREAU~ Braede_1007dcde-cd78-447d-99f7-823d82eec545_LTM1norm_annotations.mat"],
        ["GOBLELIDSTONE~_d43e27d4-4076-4150-a0d5-53eca03db13c.edf", "GOBLELIDSTONE~_d43e27d4-4076-4150-a0d5-53eca03db13c_annotations.mat"],
        ["GUZZI~ Giovann_53020603-961e-42d8-bdc5-31134ff91e87.edf", "GUZZI~ Giovann_53020603-961e-42d8-bdc5-31134ff91e87_annotations.mat"],
        ["HAMMADI~ Ayoub_bc1a1adb-d804-4bac-a07a-d0628b13d89a.edf", "HAMMADI~ Ayoub_bc1a1adb-d804-4bac-a07a-d0628b13d89a_annotations.mat"],
        ["MASAUD~ Mohamm_cc4de571-ea29-4a8f-b4b2-14047db7eb5e.edf", "MASAUD~ Mohamm_cc4de571-ea29-4a8f-b4b2-14047db7eb5e_annotations.mat"],
        ["McMILLAN~ Sydn_6dfb44c7-36ce-48ad-90cc-d967ffb824dc.edf", "McMILLAN~ Sydn_6dfb44c7-36ce-48ad-90cc-d967ffb824dc_annotations.mat"],
        ["MILLS~ Abigail_cebcb472-3af5-42d3-b688-efb49c01e58f.edf", "MILLS~ Abigail_cebcb472-3af5-42d3-b688-efb49c01e58f_annotations.mat"],
        ["MOUL~ Ashley_92889ca8-3e6c-40a0-8628-624d8889d709.edf", "MOUL~ Ashley_92889ca8-3e6c-40a0-8628-624d8889d709_annotations.mat"],
        ["PEZDERIC~ Kier_2333d768-06a9-4730-9900-dcd3381df60d.edf", "PEZDERIC~ Kier_2333d768-06a9-4730-9900-dcd3381df60d_annotations.mat"],
        ["PRICE~ Weston_7bd004d0-e2b6-42d3-b1b3-5126788b7918.edf", "PRICE~ Weston_7bd004d0-e2b6-42d3-b1b3-5126788b7918_annotations.mat"],
        ["SINJAR~ Majd_286a1fc0-7679-4785-81d7-2a0f1ec1a11e.edf", "SINJAR~ Majd_286a1fc0-7679-4785-81d7-2a0f1ec1a11e_annotations.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    
    def Maggi_Stroke_Data_3(self):
        dataset_name = "Maggi_Stroke_Data_3"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_3/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/EEGs_Whole/Maggi_Stroke_Data_3/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["ALI~ Ume_00cb8b62-12e2-4f78-8170-79f64fb15079.edf","ALI~ Ume_00cb8b62-12e2-4f78-8170-79f64fb15079_24_6_2024_annotations.mat"],
        ["AMBROSIO~ Kyle_cat2!!.edf","AMBROSIO~ Kyle_cat2!!_annotations.mat"],
        ["AUBE~ Sophia_5936864d-e163-4a56-afd5-45fe1003e138.edf","AUBE~ Sophia_5936864d-e163-4a56-afd5-45fe1003e138_annotations.mat"],
        ["BHATTI~ Arjun_better quality.edf","BHATTI~ Arjun_better quality_2_annotations.mat"],
        ["DHILLON~ Gurma_9e9c553e-46c9-4733-99c6-f2bb30dd95ba.edf","DHILLON~ Gurma_annotations.mat"],
        ["ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6.edf","ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotations.mat"],
        ["GATES~ Rylan_ecead51a-1352-4f42-bbdb-af0d041f3372.edf","GATES~ Rylan_ecead51a-1352-4f42-bbdb-af0d041f3372_annotations.mat"],
        ["HARTNETT~ Juli_558d4ced-f4ad-44f5-92db-c6364d7d9a1d.edf","HARTNETT~ Juli_558d4ced-f4ad-44f5-92db-c6364d7d9a1d_annotations.mat"],
        ["HILDEBRANDT~ C_489e2278-08f6-4b6c-ab50-65a76042c714.edf","HILDEBRANDT~ C_489e2278-08f6-4b6c-ab50-65a76042c714_annotations.mat"],
        ["LUNN~ Madison_cat2.edf","LUNN~ Madison_cat2_annotations.mat"],
        ["MUSCEDERE~ Sor_6d48a910-f98a-44e6-bbf7-dbbe5a8e0252.edf","MUSCEDERE~ Sor_6d48a910-f98a-44e6-bbf7-dbbe5a8e0252_annotations.mat"],
        ["NSOUM~ Denise_cd854db2-a82a-4aa8-9b36-aeedd42b3e29.edf","NSOUM~ Denise_cd854db2-a82a-4aa8-9b36-aeedd42b3e29_annotations.mat"],
        ["PARRAPENA~ Dan_b3944205-b98d-43c1-a6ad-09c1cb8462a1.edf","PARRAPENA~ Dan_b3944205-b98d-43c1-a6ad-09c1cb8462a1_annotations.mat"],
        ["WHITTY~ Rachel_72648762-a94d-4088-9bb1-3608b753851c.edf","WHITTY~ Rachel_72648762-a94d-4088-9bb1-3608b753851c_annotations.mat"],
        ]

        return dataset_name, data_path, eeg_files_info

    def Minette_AED_clipped(self):
        dataset_name = "Minette_AED"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Minette_AED/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/EEGs_Whole/Minette_AED/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BENNETT~ Nicol_e7d45741-4045-4aa8-983a-205275eb1d7bmar2023.edf", "BENNETT~ Nicol_e7d45741-4045-4aa8-983a-205275eb1d7bmar2023_24_7_2023__2_57_4_annotations.mat"],
        ["BENNETT~ Nicol_ea448b98-6869-4e7a-86bd-14a44952098daug2021.edf", "BENNETT~ Nicol_ea448b98-6869-4e7a-86bd-14a44952098daug2021_1_8_2023__12_22_21_annotations.mat"],
        ["FRASER~ Nealen_899657ff-f3ff-41eb-8561-4eb4e80ee6b5mar2023.edf", "FRASER~ Nealen_899657ff-f3ff-41eb-8561-4eb4e80ee6b5mar2023_8_8_2023__16_53_21_annotations.mat"],
        ["FRASER~ Nealen_8f7bb0c0-26be-40b8-b6e2-8a4de5aec013feb2022.edf", "FRASER~ Nealen_8f7bb0c0-26be-40b8-b6e2-8a4de5aec013feb2022_22_9_2023__9_18_57_annotations.mat"],
        ["FRIESEN~ Dawso_99b634bc-ce27-4d83-936a-7e6f9c8407cb.edf", "FRIESEN~ Dawso_99b634bc-ce27-4d83-936a-7e6f9c8407cb_18_9_2023__15_19_29_annotations.mat"],
        ["FRIESEN~ Dawso_de7d355d-47f9-4a2f-a97f-d5287e615c8f.edf", "FRIESEN~ Dawso_de7d355d-47f9-4a2f-a97f-d5287e615c8f_18_9_2023__18_10_10_annotations.mat"],
        ["GAETZ_ Phoebe_40bab3e7-b02f-4704-ae3e-b67a5af3130c.edf", "GAETZ_ Phoebe_40bab3e7-b02f-4704-ae3e-b67a5af3130c_21_7_2023__5_59_35_annotations.mat"],
        ["GAETZ~ Phoebe_ca90e0b3-3763-41c1-9a52-308618efd77e.edf", "GAETZ~ Phoebe_ca90e0b3-3763-41c1-9a52-308618efd77e_27_5_2023__16_1_29_annotations.mat"],
        ["HONEYWOOD_ Cha_0c5008f8-5d1d-43d5-89cf-f815f75aae97.edf", "HONEYWOOD_ Cha_0c5008f8-5d1d-43d5-89cf-f815f75aae97_19_9_2023__15_56_9_annotations.mat"],
        ["HONEYWOOD_ Cha_e3de88cc-84cc-40ae-ba45-504bfb148760.edf", "HONEYWOOD_ Cha_e3de88cc-84cc-40ae-ba45-504bfb148760_22_9_2023__8_49_29_annotations.mat"],
        ["LONGARD~ Callu_1e8e25e0-8f2d-4de1-8422-d6cf88547fb8june2022.edf", "LONGARD~ Callu_1e8e25e0-8f2d-4de1-8422-d6cf88547fb8june2022_10_8_2023__12_38_51_annotations.mat"],
        ["LONGARD~ Callu_ce820813-d31c-4afe-9395-2d424bb58444feb2023.edf", "LONGARD~ Callu_ce820813-d31c-4afe-9395-2d424bb58444feb2023_13_8_2023__15_21_3_annotations.mat"],
        ["MEIER~ Jenna_43dd9467-aa86-4599-916c-4c43448ca19b.edf", "MEIER~ Jenna_43dd9467-aa86-4599-916c-4c43448ca19b_7_5_2023__18_48_7_annotations.mat"],
        ["MEIER~ Jenna_872fb9c8-cce6-4ebb-88f8-5dc473fa320a.edf", "MEIER~ Jenna_872fb9c8-cce6-4ebb-88f8-5dc473fa320a_4_5_2023__17_50_7_annotations.mat"],
        ["MOFFET~ Cooper_38fbd417-2f6f-48fb-9b2c-55ace040e6cb.edf", "MOFFET~ Cooper_38fbd417-2f6f-48fb-9b2c-55ace040e6cb_28_4_2023__17_39_18_annotations.mat"],
        ["MOFFET~ Cooper_e8b864b9-8856-4195-b0d7-6c38040534c7.edf", "MOFFET~ Cooper_e8b864b9-8856-4195-b0d7-6c38040534c7_18_4_2023__12_27_52_annotations.mat"],
        ["MONAGHAN~ Oliv_08d952de-7597-48a1-a44a-6e7a1c9cd219May2023.edf", "MONAGHAN~ Oliv_08d952de-7597-48a1-a44a-6e7a1c9cd219May2023_14_8_2023__10_38_41_annotations.mat"],
        ["MONAGHAN~ Oliv_dba22cb3-1a13-488e-b901-9eb508919918oct2021.edf", "MONAGHAN~ Oliv_dba22cb3-1a13-488e-b901-9eb508919918oct2021_14_8_2023__16_12_36_annotations.mat"],
        ["OBOYLE~ Holly_e013d0bf-e2c0-4782-8733-fb34dd0a1984april2022.edf", "OBOYLE~ Holly_e013d0bf-e2c0-4782-8733-fb34dd0a1984april2022_17_8_2023__11_30_3_annotations.mat"],
        ["OBOYLE~ Holly_e42b10a8-c4db-41f9-9f40-a3d4fe788714May 3, 2023.edf", "OBOYLE~ Holly_e42b10a8-c4db-41f9-9f40-a3d4fe788714May 3, 2023_20_8_2023__11_53_19_annotations.mat"],
        ["OKEKE~ Oge_92a42279-6045-43d6-a2e0-c61c66463b04apr2021.edf", "OKEKE~ Oge_92a42279-6045-43d6-a2e0-c61c66463b04apr2021_24_8_2023__15_27_3_annotations.mat"],
        ["OKEKE~ Oge_ab74fbce-5f09-4a84-bc90-f1de604549ecjune2023.edf", "OKEKE~ Oge_ab74fbce-5f09-4a84-bc90-f1de604549ecjune2023(1)_31_8_2023__14_27_24_annotations.mat"],
        ["PALLARESPULIDO_9d9e9b46-a26f-46fe-b360-39dfca75c372may2023.edf", "PALLARESPULIDO_9d9e9b46-a26f-46fe-b360-39dfca75c372may2023_29_8_2023__10_47_34_annotations.mat"],
        ["PALLARESPULIDO_f6f10281-d5f0-4e05-afd5-78db01268666nov2021.edf", "PALLARESPULIDO_f6f10281-d5f0-4e05-afd5-78db01268666nov2021_30_8_2023__14_41_34_annotations.mat"],
        ["POWELL~ Morgan_51e4ebea-fd43-4884-a8eb-48f1bcf4d2a5april2023.edf", "POWELL~ Morgan_51e4ebea-fd43-4884-a8eb-48f1bcf4d2a5april2023_5_9_2023__10_18_3_annotations.mat"],
        ["POWELL~ Morgan_66c348de-93aa-43a9-b91a-9c8a978a4f5fsept2022.edf", "POWELL~ Morgan_66c348de-93aa-43a9-b91a-9c8a978a4f5fsept2022_7_9_2023__13_2_8_annotations.mat"],
        ["QUINTON~ Morga_23e9f290-1010-4669-a90f-87c458daffad.edf", "QUINTON~ Morga_23e9f290-1010-4669-a90f-87c458daffad_3_4_2023__13_34_3_annotations.mat"],
        ["QUINTON~ Morga_651e748b-616f-4741-858a-75955aae8968.edf", "QUINTON~ Morga_651e748b-616f-4741-858a-75955aae8968_31_3_2023__15_24_16_annotations.mat"],
        ["RITSON-BENNET~_180f1128-850b-4769-a9fe-4145d0b1a3ea.edf", "RITSON-BENNET~_180f1128-850b-4769-a9fe-4145d0b1a3ea_31_3_2023__9_53_4_annotations.mat"],
        ["RITSON-BENNETT_022648e5-617e-46fc-b974-22f9c441808f.edf", "RITSON-BENNETT_022648e5-617e-46fc-b974-22f9c441808f_29_3_2023__16_48_33_annotations.mat"],
        ["SABADO_ Marek_af6286cd-0f6a-4055-b69f-1f32bc66477b.edf", "SABADO_ Marek_af6286cd-0f6a-4055-b69f-1f32bc66477b (1)_6_3_2023__14_56_35_annotations.mat"],
        ["SABADO~ Marek_df512d13-ca61-4c54-af6a-0f25b25fd8fe.edf", "SABADO~ Marek_df512d13-ca61-4c54-af6a-0f25b25fd8fe_20_3_2023__8_41_50_annotations.mat"],
        ["STOCKMAN~ TYLE_3ed33d0e-ba4a-4969-b586-86fa9b551131.edf", "STOCKMAN~ TYLE_3ed33d0e-ba4a-4969-b586-86fa9b551131_3_3_2023__18_11_59_annotations.mat"],
        ["STOCKMAN~ TYLE_b279967e-2ef3-425a-8d7e-03642c7fc042.edf", "STOCKMAN~ TYLE_b279967e-2ef3-425a-8d7e-03642c7fc042_13_2_2023__11_45_35_annotations.mat"],
        ["West_Parker-mar-2023_3059749451.edf", "West Parker-mar-2023_3059749451_11_9_2023__11_59_32_annotations.mat"],
        ["WEST~ Parker_9ee4fbb9-0196-47a1-91ac-ba707f346447April2022.edf", "WEST~ Parker_9ee4fbb9-0196-47a1-91ac-ba707f346447April2022_14_9_2023__16_23_39_annotations.mat"],
        ["WILDEMAN~ Taiv_45c8bf5e-1200-42bb-bfb8-c4f08c6efa7c.edf", "WILDEMAN~ Taiv_45c8bf5e-1200-42bb-bfb8-c4f08c6efa7c_20_9_2023__23_2_43_annotations.mat"],
        ["WILDEMAN~ Taiv_e8db78f1-17c7-4820-ac0f-7562bfe19a97.edf", "WILDEMAN~ Taiv_e8db78f1-17c7-4820-ac0f-7562bfe19a97_21_9_2023__17_37_10_annotations.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    def Multidetect_Validated_HFO_clipped(self):
        dataset_name = "Multidetect_Validated_HFO"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Multidetect_Maggi_Validated_Zurich_Pats/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/EEGs_Whole/Multidetect_Maggi_Validated_Zurich_Pats/"
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BONERTZ~ Denve_f651bec5-8213-4df8-a500-e5cde80eed6d.edf", "BONERTZ~ Denve_f651bec5-8213-4df8-a500-e5cde80eed6d_annotations.mat"],
        ["BRAVO~ Isabell_dcda0647-7aba-49f7-a403-6677416e3e93.edf", "BRAVO~ Isabell_dcda0647-7aba-49f7-a403-6677416e3e93_annotations.mat"],
        ["BRIGGSRIGIO~ S_f4d226da-a40f-4ab6-91fa-cf8290227e22.edf", "BRIGGSRIGIO~ S_f4d226da-a40f-4ab6-91fa-cf8290227e22_annotations.mat"],
        ["CONTRERAS~ Hol_b747da66-1ae0-4f27-bbf8-ce6e92160a52.edf", "CONTRERAS~ Hol_b747da66-1ae0-4f27-bbf8-ce6e92160a52_annotations.mat"],
        ["DEDRICK~ Olivi_412b3f13-ebc6-4ed5-bc2d-792450c72e84.edf", "DEDRICK~ Olivi_412b3f13-ebc6-4ed5-bc2d-792450c72e84_annotations.mat"],
        ["DEVER~ Rory_db8f3b9b-90cf-4ad7-a253-2f28d798fc05.edf", "DEVER~ Rory_db8f3b9b-90cf-4ad7-a253-2f28d798fc05_annotations.mat"],
        ["FREI~ Domenica_9c81500d-872f-42c6-a0ee-d6d8f4c23e2f.edf", "FREI~ Domenica_9c81500d-872f-42c6-a0ee-d6d8f4c23e2f_annotations.mat"],
        ["GOBLELIDSTONE~_1f68f041-84ac-41b0-be3d-638d17500c7a.edf", "GOBLELIDSTONE~_1f68f041-84ac-41b0-be3d-638d17500c7a_annotations.mat"],
        ["HARGATE~ Caitl_043600c6-d543-481f-9787-e15964cc46ab.edf", "HARGATE~ Caitl_043600c6-d543-481f-9787-e15964cc46ab_annotations.mat"],
        ["HOULT~ Liam_ee1c1462-4645-4cbc-8a8c-5377b75305e6.edf", "HOULT~ Liam_ee1c1462-4645-4cbc-8a8c-5377b75305e6_annotations.mat"],
        ["MABIOR~ Diing_086c6be5-b0a5-4093-8d87-3cdd7e22a966.edf", "MABIOR~ Diing_086c6be5-b0a5-4093-8d87-3cdd7e22a966_annotations.mat"],
        ["MACGREGOR~ Sop_6d81a4d7-a9e5-48dd-aed6-b16e3c30cb18.edf", "MACGREGOR~ Sop_6d81a4d7-a9e5-48dd-aed6-b16e3c30cb18_annotations.mat"],
        ["MARTIN~ Scott_613958e7-a4a1-44c6-942c-7110b68bc6da.edf", "MARTIN~ Scott_613958e7-a4a1-44c6-942c-7110b68bc6da_annotations.mat"],
        ["MASRI~ Malik_e71a0a85-c927-4d58-acc8-649399a953a3.edf", "MASRI~ Malik_e71a0a85-c927-4d58-acc8-649399a953a3_annotations.mat"],
        ["MATHESON~ Finl_cc1379e2-9838-4a2e-b8c2-c9f99cc4e8bc.edf", "MATHESON~ Finl_cc1379e2-9838-4a2e-b8c2-c9f99cc4e8bc_annotations.mat"],
        ["McELLIGOTT~ Ev_400b9479-5a48-4712-abd6-7b9ef449881b.edf", "McELLIGOTT~ Ev_400b9479-5a48-4712-abd6-7b9ef449881b_annotations.mat"],
        ["OKEEFFE~ Emma_4595d807-92d7-4cb7-a36a-bfc1df8afd24.edf", "OKEEFFE~ Emma_4595d807-92d7-4cb7-a36a-bfc1df8afd24_annotations.mat"],
        ["OLSON~ Andrew_16027753-5808-4498-aed8-966f29a04a04.edf", "OLSON~ Andrew_16027753-5808-4498-aed8-966f29a04a04_annotations.mat"],
        ["RAMIREZ~ Amand_91bd3985-137c-405d-a532-b403f7c94705.edf", "RAMIREZ~ Amand_91bd3985-137c-405d-a532-b403f7c94705_annotations.mat"],
        ["SALMON~ Olivia_7e208bf8-76e4-4217-a472-5e2de8b92baf.edf", "SALMON~ Olivia_7e208bf8-76e4-4217-a472-5e2de8b92baf_annotations.mat"],
        ["SINJAR~ Majd_c7c32ec6-597b-494b-b451-ad6960720c98.edf", "SINJAR~ Majd_c7c32ec6-597b-494b-b451-ad6960720c98_annotations.mat"],
        ["TORRIE~ Brookl_50ddff17-9579-4b28-874f-756be76b6c19.edf", "TORRIE~ Brookl_50ddff17-9579-4b28-874f-756be76b6c19_annotations.mat"],
        ["WASYLENKA~ Dyl_0f73ce29-1914-4169-98d7-4613433ac73e.edf", "WASYLENKA~ Dyl_0f73ce29-1914-4169-98d7-4613433ac73e_annotations.mat"],
        ["WHALEN~ Riley_8780cda3-ccb4-4918-8c25-3f6f73798886.edf", "WHALEN~ Riley_8780cda3-ccb4-4918-8c25-3f6f73798886_annotations.mat"],
        ["WICKHORST~ Jos_a67e52c8-45ab-4a94-9550-66eef67c9c90.edf", "WICKHORST~ Jos_a67e52c8-45ab-4a94-9550-66eef67c9c90_annotations.mat"],
        ["WILSON~ Jack_973d043c-c443-4c62-8bfc-07697e98951d.edf", "WILSON~ Jack_973d043c-c443-4c62-8bfc-07697e98951d_annotations.mat"],
        ["WOOD~ Dawson_a6c0af5a-b1c7-47fe-8638-e4199ae768d2.edf", "WOOD~ Dawson_a6c0af5a-b1c7-47fe-8638-e4199ae768d2_annotations.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    def Ree_11Pats_Validation_July2024(self):
        dataset_name = "Ree_11Pats_Validation_July2024"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Ree_Files_July_2024/"
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Ree_Validation/"
        elif socket.gethostname() == "DLP":
            #data_path = "E:/Ree_Files_July_2024/"
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Ree_Validation/"
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALI~ Ziya_Clipped.edf","ALI~ Ziya_Clipped_26_8_2024__16_0_41_annotations.mat", "ALI~ Ziya_Clipped_overcomplete_HFO_Marks.mat"],
        ["AUBE~ Lincoln_Clipped.edf","AUBE~ Lincoln_Clipped_26_8_2024__16_51_8_annotations.mat", "AUBE~ Lincoln_Clipped_overcomplete_HFO_Marks.mat"],
        ["CAMARA~ Isla_Clipped.edf","CAMARA~ Isla_Clipped_26_8_2024__17_6_53_annotations.mat", "CAMARA~ Isla_Clipped_overcomplete_HFO_Marks.mat"],
        ["ELRAFIH~ Musta_Clipped.edf","ELRAFIH~ Musta_Clipped_26_8_2024__21_51_2_annotations.mat", "ELRAFIH~ Musta_Clipped_overcomplete_HFO_Marks.mat"],
        ["HASSANIN~ Yasmine_Clipped.edf","HASSANIN~ Yasmine_Clipped_26_8_2024__22_16_2_annotations.mat", "HASSANIN~ Yasmine_Clipped_overcomplete_HFO_Marks.mat"],
        ["LACOMBE~ Luke_Clipped.edf","LACOMBE~ Luke_Clipped_26_8_2024__23_12_56_annotations.mat", "LACOMBE~ Luke_Clipped_overcomplete_HFO_Marks.mat"],
        ["LECERF~ William_Clipped.edf","LECERF~ William_Clipped_27_8_2024__16_50_23_annotations.mat", "LECERF~ William_Clipped_overcomplete_HFO_Marks.mat"],
        ["MOUSSA~ Noora_Clipped.edf","MOUSSA~ Noora_Clipped_27_8_2024__17_9_11_annotations.mat", "MOUSSA~ Noora_Clipped_overcomplete_HFO_Marks.mat"],
        ["ODLAND~ Ember_Clipped.edf","ODLAND~ Ember_Clipped_30_8_2024__14_50_49_annotations.mat", "ODLAND~ Ember_Clipped_overcomplete_HFO_Marks.mat"],
        ["ROSEVEAR~ George_Clipped.edf","ROSEVEAR~ George_Clipped_27_8_2024__17_34_7_annotations.mat", "ROSEVEAR~ George_Clipped_overcomplete_HFO_Marks.mat"],
        ["VANBOVEN~ Gerrit_Clipped.edf","VANBOVEN~ Gerrit_Clipped_27_8_2024__17_54_48_annotations.mat", "VANBOVEN~ Gerrit_Clipped_overcomplete_HFO_Marks.mat"],
        ]
        
        return dataset_name, data_path, eeg_files_info

    def Maggi_11Pats_Validation_Dez2024(self):
        dataset_name = "Maggi_11Pats_Validation_Dez2024"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Ree_Files_July_2024/"
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Ree_Validation/"
        elif socket.gethostname() == "DLP":
            #data_path = "E:/Ree_Files_July_2024/"
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Overcomplete_HFO_Validation/Validated_Files_Maggi_Dez2024/"
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALI~ Ziya_Clipped.edf","ALI~ Ziya_Clipped_annotations corrected.mat", "ALI~ Ziya_Clipped_overcomplete_HFO_candidates.mat", "ALI~ Ziya_Clipped_dlpHFO_Marks.mat", "ALI~ Ziya_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["AUBE~ Lincoln_Clipped.edf","AUBE~ Lincoln_Clipped_annotations_corrected.mat", "AUBE~ Lincoln_Clipped_overcomplete_HFO_candidates.mat", "AUBE~ Lincoln_Clipped_dlpHFO_Marks.mat", "AUBE~ Lincoln_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["CAMARA~ Isla_Clipped.edf","CAMARA~ Isla_Clipped_annotations corrected.mat", "CAMARA~ Isla_Clipped_overcomplete_HFO_candidates.mat", "CAMARA~ Isla_Clipped_dlpHFO_Marks.mat", "CAMARA~ Isla_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ELRAFIH~ Musta_Clipped.edf","ELRAFIH~ Musta_Clipped_corrected.mat", "ELRAFIH~ Musta_Clipped_overcomplete_HFO_candidates.mat", "ELRAFIH~ Musta_Clipped_dlpHFO_Marks.mat", "ELRAFIH~ Musta_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["HASSANIN~ Yasmine_Clipped.edf","HASSANIN~ Yasmine_Clipped_corrected.mat", "HASSANIN~ Yasmine_Clipped_overcomplete_HFO_candidates.mat", "HASSANIN~ Yasmine_Clipped_dlpHFO_Marks.mat", "HASSANIN~ Yasmine_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["LACOMBE~ Luke_Clipped.edf","LACOMBE~ Luke_corrected.mat", "LACOMBE~ Luke_Clipped_overcomplete_HFO_candidates.mat", "LACOMBE~ Luke_Clipped_dlpHFO_Marks.mat", "LACOMBE~ Luke_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["LECERF~ William_Clipped.edf","LECERF~ William_Clipped_corrected.mat", "LECERF~ William_Clipped_overcomplete_HFO_candidates.mat", "LECERF~ William_Clipped_dlpHFO_Marks.mat", "LECERF~ William_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["MOUSSA~ Noora_Clipped.edf","MOUSSA~ Noora_Clipped_corrected.mat", "MOUSSA~ Noora_Clipped_overcomplete_HFO_candidates.mat", "MOUSSA~ Noora_Clipped_dlpHFO_Marks.mat", "MOUSSA~ Noora_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ODLAND~ Ember_Clipped.edf","ODLAND~ Ember_Clipped_corrected.mat", "ODLAND~ Ember_Clipped_overcomplete_HFO_candidates.mat", "ODLAND~ Ember_Clipped_dlpHFO_Marks.mat", "ODLAND~ Ember_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ROSEVEAR~ George_Clipped.edf","ROSEVEAR~ George_Clipped_corrected.mat", "ROSEVEAR~ George_Clipped_overcomplete_HFO_candidates.mat", "ROSEVEAR~ George_Clipped_dlpHFO_Marks.mat", "ROSEVEAR~ George_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["VANBOVEN~ Gerrit_Clipped.edf","VANBOVEN~ Gerrit_Clipped_corrected.mat", "VANBOVEN~ Gerrit_Clipped_overcomplete_HFO_candidates.mat", "VANBOVEN~ Gerrit_Clipped_dlpHFO_Marks.mat", "VANBOVEN~ Gerrit_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ]
        
        return dataset_name, data_path, eeg_files_info
    
    def Maggi_11Pats_Validation_Dez2024_PostHoc(self):
        dataset_name = "Maggi_11Pats_Validation_Dez2024"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Ree_Files_July_2024/"
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Ree_Validation_PostHoc/"
        elif socket.gethostname() == "DLP":
            #data_path = "E:/Ree_Files_July_2024/"
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Overcomplete_HFO_Validation/Validated_Files_Maggi_Dez2024_PostHoc/"
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALI~ Ziya_Clipped.edf","ALI~ Ziya_Clipped_maggi_annotations_corrected.mat", "ALI~ Ziya_Clipped_overcomplete_HFO_candidates.mat", "ALI~ Ziya_Clipped_dlpHFO_Marks.mat", "ALI~ Ziya_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["AUBE~ Lincoln_Clipped.edf","AUBE~ Lincoln_Clipped_maggi_annotations_corrected.mat", "AUBE~ Lincoln_Clipped_overcomplete_HFO_candidates.mat", "AUBE~ Lincoln_Clipped_dlpHFO_Marks.mat", "AUBE~ Lincoln_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["CAMARA~ Isla_Clipped.edf","CAMARA~ Isla_Clipped_maggi_annotations_corrected.mat", "CAMARA~ Isla_Clipped_overcomplete_HFO_candidates.mat", "CAMARA~ Isla_Clipped_dlpHFO_Marks.mat", "CAMARA~ Isla_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ELRAFIH~ Musta_Clipped.edf","ELRAFIH~ Musta_Clipped_maggi_annotations_corrected.mat", "ELRAFIH~ Musta_Clipped_overcomplete_HFO_candidates.mat", "ELRAFIH~ Musta_Clipped_dlpHFO_Marks.mat", "ELRAFIH~ Musta_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["HASSANIN~ Yasmine_Clipped.edf","HASSANIN~ Yasmine_Clipped_maggi_annotations_corrected.mat", "HASSANIN~ Yasmine_Clipped_overcomplete_HFO_candidates.mat", "HASSANIN~ Yasmine_Clipped_dlpHFO_Marks.mat", "HASSANIN~ Yasmine_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["LACOMBE~ Luke_Clipped.edf","LACOMBE~ Luke_maggi_annotations_corrected.mat", "LACOMBE~ Luke_Clipped_overcomplete_HFO_candidates.mat", "LACOMBE~ Luke_Clipped_dlpHFO_Marks.mat", "LACOMBE~ Luke_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["LECERF~ William_Clipped.edf","LECERF~ William_Clipped_maggi_annotations_corrected.mat", "LECERF~ William_Clipped_overcomplete_HFO_candidates.mat", "LECERF~ William_Clipped_dlpHFO_Marks.mat", "LECERF~ William_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["MOUSSA~ Noora_Clipped.edf","MOUSSA~ Noora_Clipped_maggi_annotations_corrected.mat", "MOUSSA~ Noora_Clipped_overcomplete_HFO_candidates.mat", "MOUSSA~ Noora_Clipped_dlpHFO_Marks.mat", "MOUSSA~ Noora_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ODLAND~ Ember_Clipped.edf","ODLAND~ Ember_Clipped_maggi_annotations_corrected.mat", "ODLAND~ Ember_Clipped_overcomplete_HFO_candidates.mat", "ODLAND~ Ember_Clipped_dlpHFO_Marks.mat", "ODLAND~ Ember_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["ROSEVEAR~ George_Clipped.edf","ROSEVEAR~ George_Clipped_maggi_annotations_corrected.mat", "ROSEVEAR~ George_Clipped_overcomplete_HFO_candidates.mat", "ROSEVEAR~ George_Clipped_dlpHFO_Marks.mat", "ROSEVEAR~ George_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ["VANBOVEN~ Gerrit_Clipped.edf","VANBOVEN~ Gerrit_Clipped_maggi_annotations_corrected.mat", "VANBOVEN~ Gerrit_Clipped_overcomplete_HFO_candidates.mat", "VANBOVEN~ Gerrit_Clipped_dlpHFO_Marks.mat", "VANBOVEN~ Gerrit_Clipped_overcomplete_HFO_candidates_2025.mat"],
        ]
        
        return dataset_name, data_path, eeg_files_info

    def Overcomplete_Validated_Blobs_Physio_Patients(self):
        dataset_name = "Overcomplete_Validated_Blobs_Physio_Patients"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Ree_Files_July_2024/"
            data_path = "F:/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Overcomplete_Validated_Blobs_Physio_Patients/"
        elif socket.gethostname() == "DLP":
            #data_path = "E:/Ree_Files_July_2024/"
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Scalp_HFO_GoldStandard/EEGs_Whole/Overcomplete_Validated_Blobs_Physio_Patients/"
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALI~ Ziya_Clipped.edf","ALI~ Ziya_Clipped_maggi_annotations_corrected.mat"],
        ["AUBE~ Lincoln_Clipped.edf","AUBE~ Lincoln_Clipped_maggi_annotations_corrected.mat"],
        ["CAMARA~ Isla_Clipped.edf","CAMARA~ Isla_Clipped_maggi_annotations_corrected.mat"],
        ["ELRAFIH~ Musta_Clipped.edf","ELRAFIH~ Musta_Clipped_maggi_annotations_corrected.mat"],
        ["HASSANIN~ Yasmine_Clipped.edf","HASSANIN~ Yasmine_Clipped_maggi_annotations_corrected.mat"],
        ["LACOMBE~ Luke_Clipped.edf","LACOMBE~ Luke_maggi_annotations_corrected.mat"],
        ["LECERF~ William_Clipped.edf","LECERF~ William_Clipped_maggi_annotations_corrected.mat"],
        ["MOUSSA~ Noora_Clipped.edf","MOUSSA~ Noora_Clipped_maggi_annotations_corrected.mat"],
        ["ODLAND~ Ember_Clipped.edf","ODLAND~ Ember_Clipped_maggi_annotations_corrected.mat"],
        ["ROSEVEAR~ George_Clipped.edf","ROSEVEAR~ George_Clipped_maggi_annotations_corrected.mat"],
        ["VANBOVEN~ Gerrit_Clipped.edf","VANBOVEN~ Gerrit_Clipped_maggi_annotations_corrected.mat"],
        ]
        
        return dataset_name, data_path, eeg_files_info

    def HFOHealthy1monto2yrs(self):
        dataset_name = "HFOHealthy1monto2yrs"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Clipped_EDFs/HFOHealthy1monto2yrs/"
        elif socket.gethostname() == "DLP":
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Ree_Big_Data_Transfer/HFOHealthy1monto2yrs/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALAIN~ Lennox_Clipped.edf", "_"],
        ["ALI~ Ziya_Clipped.edf", "_"],
        ["ANDERSON~ Oliver_Clipped.edf", "_"],
        ["AUBE~ Lincoln_Clipped.edf", "_"],
        ["BAH~ Mamoudou_Clipped.edf", "_"],
        ["BATSTONE~ Madeline_Clipped.edf", "_"],
        ["BOBIER~ Charle_Clipped.edf", "_"],
        ["BURTON~ Lincol_Clipped.edf", "_"],
        ["BYUN~ Naomi_Clipped.edf", "_"],
        ["CALLANDER~ Fem_Clipped.edf", "_"],
        ["CAMARA~ Isla_Clipped.edf", "_"],
        ["CANTUBA~ Elysian_Clipped.edf", "_"],
        ["CASE~ Charlotte_Clipped.edf", "_"],
        ["CHEN~ Erick_Clipped.edf", "_"],
        ["CLARK~ Dean_Clipped.edf", "_"],
        ["CYR~ Sophie-Eloise_Clipped.edf", "_"],
        ["DOMINGUEZ-CANCHICA_Eva_Clipped.edf", "_"],
        ["EHSSAN~ Samah_Clipped.edf", "_"],
        ["ELRAFIH~ Musta_Clipped.edf", "_"],
        ["FAROOQUI~ Hashir_Clipped.edf", "_"],
        ["HASHMI~ Ayrah_Clipped.edf", "_"],
        ["HASSANIN~ Yasmine_Clipped.edf", "_"],
        ["HIEBERT~ Hunter_Clipped.edf", "_"],
        ["ISKAUSKAS~ Atticus_Clipped.edf", "_"],
        ["KAUR~ Aleena_Clipped.edf", "_"],
        ["LACOMBE~ Luke_Clipped.edf", "_"],
        ["LAROCHELLE~ Aribella_Clipped.edf", "_"],
        ["LECERF~ William_Clipped.edf", "_"],
        ["MILLICHAMP~ Orion_Clipped.edf", "_"],
        ["MOJICA~ Anton_Clipped.edf", "_"],
        ["MOUSSA~ Noora_Clipped.edf", "_"],
        ["ODLAND~ Ember_Clipped.edf", "_"],
        ["PERRIGO~ Elizabeth_Clipped.edf", "_"],
        ["RAMOS~ Katalin_Clipped.edf", "_"],
        ["ROBERTS~ Salem_Clipped.edf", "_"],
        ["ROBERTSON~ Red_Clipped.edf", "_"],
        ["ROCSKAR~ Stevie_Clipped.edf", "_"],
        ["ROSEVEAR~ George_Clipped.edf", "_"],
        ["ROWETT~ Alicia_Clipped.edf", "_"],
        ["SECORD~ Damien_Clipped.edf", "_"],
        ["SEJDIC~ Melia_Clipped.edf", "_"],
        ["SMITH~ Toby_Clipped.edf", "_"],
        ["SNIDER~ Victor_Clipped.edf", "_"],
        ["SPEARINGRUIZ~Judah _Clipped.edf", "_"],
        ["STEWART~ William_Clipped.edf", "_"],
        ["ST-DENIS~ William_Clipped.edf", "_"],
        ["THIESSEN~ Carter_Clipped.edf", "_"],
        ["TOMAS~ Isabela_Clipped.edf", "_"],
        ["ULRICH~ Shaya_Clipped.edf", "_"],
        ["VANBOVEN~ Gerrit_Clipped.edf", "_"],
        ["WHITEWAY~ Female(Mia)_Clipped.edf", "_"],
        ["WILFORT~ Edison_Clipped.edf", "_"],
        ]

        return dataset_name, data_path, eeg_files_info

    def HFOHealthy3to5yrs(self):
        dataset_name = "HFOHealthy3to5yrs"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Clipped_EDFs/HFOHealthy3to5yrs/"
        elif socket.gethostname() == "DLP":
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Ree_Big_Data_Transfer/HFOHealthy3to5yrs/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["AKDENIZ~ Yaren_Clipped.edf", "_"],
        ["BATCHELOR~ Theo_Clipped.edf", "_"],
        ["BREMNER~ Lola_Clipped.edf", "_"],
        ["BRIDEAU~ Laine_Clipped.edf", "_"],
        ["COUTURE~ Conner_Clipped.edf", "_"],
        ["CZAJKOWSKI~ Alexander_Clipped.edf", "_"],
        ["GRAHAM~ Bennet_Clipped.edf", "_"],
        ["GUY~ Madison_Clipped.edf", "_"],
        ["GUZAMN~ Dominic_Clipped.edf", "_"],
        ["HAHN~ Hadley_Clipped.edf", "_"],
        ["HANSEN~ Jacob_Clipped.edf", "_"],
        ["LANE~ Gabriel_Clipped.edf", "_"],
        ["LANG~ Hugo_Clipped.edf", "_"],
        ["LECLERC~ Noah_Clipped.edf", "_"],
        ["McKEEVER~ Christian_Clipped.edf", "_"],
        ["MOOJELSKY~ Addison_Clipped.edf", "_"],
        ["MURPHY-GREEN~ _Walter_Clipped.edf", "_"],
        ["NEDHAM~ Jackson_Clipped.edf", "_"],
        ["OUYANG~ Alexander_Clipped.edf", "_"],
        ["PATTERSON~ Wyatt_Clipped.edf", "_"],
        ["PIAYDA~ Theodor_Clipped.edf", "_"],
        ["PIOSCA~ Calvin_Clipped.edf", "_"],
        ["RENN~ Max_Clipped.edf", "_"],
        ["ROY~ Colton_Clipped.edf", "_"],
        ["SAMRA~ Harman_Clipped.edf", "_"],
        ["VIRK~ Samarbir_Clipped.edf", "_"],
        ["WILLIAMS~ Reese_Clipped.edf", "_"],
        ["WILSON~ Estelle_Clipped.edf", "_"],
        ["ADE-OKESOLA~ Grace_Clipped.edf",""],
        ["BARRERA~ Emilia_Clipped.edf",""],
        ["BELLO~ Eveshorema_Clipped.edf",""],
        ["BROWN~Meisha_Clipped.edf",""],
        ["HERRERA~ Franzlyne_Clipped.edf",""],
        ["JONES~ Emma_Clipped.edf",""],
        ["KENNEDY~Ariella_Clipped.edf",""],
        ["KIM~ Jenny_Clipped.edf",""],
        ["LAVOIE~ Zara_Clipped.edf",""],
        ["MARTIN~ Scarlett_Clipped.edf",""],
        ["MCINNIS~ Aurora_Clipped.edf",""],
        ["REYES~ Cherese_Clipped.edf",""],
        ["SONI~ Vrisha_Clipped.edf",""],
        ]

        return dataset_name, data_path, eeg_files_info
    
    def HFOHealthy6to10yrs(self):
        dataset_name = "HFOHealthy6to10yrs"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Clipped_EDFs/HFOHealthy6to10yrs/"
        elif socket.gethostname() == "DLP":
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Ree_Big_Data_Transfer/HFOHealthy6to10yrs/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ANTHONY-MALONE_Kinley_Clipped.edf", "_"],
        ["BALCHA~ Japheth_Clipped.edf", "_"],
        ["BANSRAO~ Gurshaan_Clipped.edf", "_"],
        ["BARDEN~ Nixon_Clipped.edf", "_"],
        ["BERTIN~ Shanika-Joyce_Clipped.edf", "_"],
        ["BROGER~ Alexander_Clipped.edf", "_"],
        ["CARR~ Abby_Clipped.edf", "_"],
        ["CASSAM~ Miles_Clipped.edf", "_"],
        ["CHARCHUN~ Ayla_Clipped.edf", "_"],
        ["CULL~ Leah_Clipped.edf", "_"],
        ["CYRE~ Riley_Clipped.edf", "_"],
        ["DEKUYPER~ Esther_Clipped.edf", "_"],
        ["DELEEUW~ Titus_Clipped.edf", "_"],
        ["DERIX~ Luke_Clipped.edf", "_"],
        ["DEY~ Xavi_Clipped.edf", "_"],
        ["DICK~ Sage_Clipped.edf", "_"],
        ["DRAKE~ Gabriel_Clipped.edf", "_"],
        ["DRYSDALE~ Findley_Clipped.edf", "_"],
        ["GERRARD~ Payton_Clipped.edf", "_"],
        ["GRAHAM~ Peter_Clipped.edf", "_"],
        ["HANSFORD-WELKE_Wyatt_Clipped.edf", "_"],
        ["HOLLA~ Mythri_Clipped.edf", "_"],
        ["KAISER~ Jack_Clipped.edf", "_"],
        ["KAMENKA~ Gavin_Clipped.edf", "_"],
        ["KEBEDE~ Saron_Clipped.edf", "_"],
        ["KNOWLTON-TENEY_Owen_Clipped.edf", "_"],
        ["KRISA~ Hailey_Clipped.edf", "_"],
        ["LANDAVERDE~ Xandro_Clipped.edf", "_"],
        ["LAWSON~ Harlow_Clipped.edf", "_"],
        ["MARTIN~ John_Clipped.edf", "_"],
        ["MATHISONWOOLLE_Kaiah_Clipped.edf", "_"],
        ["MCDOWELL~ Emery_Clipped.edf", "_"],
        ["MING~ Jackson_Clipped.edf", "_"],
        ["PICKARD~ Emma_Clipped.edf", "_"],
        ["POIRIER~ Preston_Clipped.edf", "_"],
        ["PONOMARJOVA~ Gabriella_Clipped.edf", "_"],
        ["RICHARD~ Noah_Clipped.edf", "_"],
        ["ROCHETTA~ Zoey_Clipped.edf", "_"],
        ["ROWLEY~ Cylus_Clipped.edf", "_"],
        ["SAULTEAUX~ Kingston_Clipped.edf", "_"],
        ["SIDOCK~ Michael_Clipped.edf", "_"],
        ["SKULMOSKI~ Logan_Clipped.edf", "_"],
        ["TASKINEN~ Reagan_Clipped.edf", "_"],
        ["TAYLOR~ Marilyn_Clipped.edf", "_"],
        ["VANDENBROEK~ Blake_Clipped.edf", "_"],
        ["WITHELL~ Grace_Clipped.edf", "_"],
        ["WOLFEAR~ Raynah_Clipped.edf", "_"],
        ["YOUNG~ Abigail_Clipped.edf", "_"],
        ["WHITE~ Maddox_Clipped.edf", "_"],
        ["CLARKE~ Onjali_Clipped.edf", "_"],
        ["FRIESEN-BOWMAN_Taliah_Clipped.edf", "_"],
        ]

        return dataset_name, data_path, eeg_files_info

    def HFOHealthy11to13yrs(self):
        dataset_name = "HFOHealthy11to13yrs"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Clipped_EDFs/HFOHealthy11to13yrs/"
        elif socket.gethostname() == "DLP":
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Ree_Big_Data_Transfer/HFOHealthy11to13yrs/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ABOELAZAB~ Joudi_Clipped.edf", "_"],
        ["ALIYEVA~ Emiliya_Clipped.edf", "_"],
        ["BLAIR-HALMAZNA_Grady_Clipped.edf", "_"],
        ["BRADY~ Mitchel_Clipped.edf", "_"],
        ["CLARK~ Jasper_Clipped.edf", "_"],
        ["DORMA~ Kimberly_Clipped.edf", "_"],
        ["FLEETON~ Isabelle_Clipped.edf", "_"],
        ["GIROUX~ Madeleine_Clipped.edf", "_"],
        ["HARRIS-STILL~Kellan_Clipped.edf", "_"],
        ["HAUER~ Kassidy_Clipped.edf", "_"],
        ["HENDY~ Brianna_Clipped.edf", "_"],
        ["HUGHES~ Zackery_Clipped.edf", "_"],
        ["IBARRARODRIQUE_Hugo_Clipped.edf", "_"],
        ["IDISI~ Oghenefejiro_Clipped.edf", "_"],
        ["JESSOME-WEISS~_Archer_Clipped.edf", "_"],
        ["KAPILA~ Jai_Clipped.edf", "_"],
        ["KASONGO~ Abigael_Clipped.edf", "_"],
        ["KILLEN~ Ayden_Clipped.edf", "_"],
        ["KLEPAYCHUK~ Joel_Clipped.edf", "_"],
        ["KRAVCHENKO~ Avery_Clipped.edf", "_"],
        ["MA~ Janet_Clipped.edf", "_"],
        ["MACKERETH-BAKER_Michael_Clipped.edf", "_"],
        ["MARSHALL~ Ethan_Clipped.edf", "_"],
        ["MARTIN~ Avery_Clipped.edf", "_"],
        ["MATHARU~ Harkirat_Clipped.edf", "_"],
        ["McDOUGALL~ Liam_Clipped.edf", "_"],
        ["MCFADYEN~ James_Clipped.edf", "_"],
        ["McLAINE~ Sophie_Clipped.edf", "_"],
        ["MEYER~ Halle_Clipped.edf", "_"],
        ["MURPHY~ Kaleigh_Clipped.edf", "_"],
        ["NOVELLO~ Santino_Clipped.edf", "_"],
        ["OCHOA~ David_Clipped.edf", "_"],
        ["PANGILINAN~ Nate_Clipped.edf", "_"],
        ["PARK~ Alyssa_Clipped.edf", "_"],
        ["RAVAL~ Stuti_Clipped.edf", "_"],
        ["SANDERSON~ Kale_Clipped.edf", "_"],
        ["SCOTT~ Ethan_Clipped.edf", "_"],
        ["SHIELDS~ Shae-Ann_Clipped.edf", "_"],
        ["URQUHART~ Bethany_Clipped.edf", "_"],
        ["WILLIS~ Drew_Clipped.edf", "_"],
        ["YIP~ Claudia_Clipped.edf", "_"],
        ["ZIMOLA~ Benjamin_Clipped.edf", "_"],
        ["ADAMS~ Teagan_Clipped.edf", "_"],
        #["JONES_Hadley_Clipped.edf", "_"],
        ]
        
        return dataset_name, data_path, eeg_files_info
   
    def HFOHealthy14to17yrs(self):
        dataset_name = "HFOHealthy14to17yrs"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Physio_HFO_Clipped_EDFs/HFOHealthy14to17yrs/"
        elif socket.gethostname() == "DLP":
            data_path = "C:/Users/HFO/Documents/Postdoc_Calgary/Research/Physiological_HFO/Ree_Big_Data_Transfer/HFOHealthy14to17yrs/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ARCHAMBAULT~ Brook-Lynn_Clipped.edf", "_"],
        ["ARNOTT~ Grace_Clipped.edf", "_"],
        ["AWAN~ Adel_Clipped.edf", "_"],
        ["BADE~ Gracie_Clipped.edf", "_"],
        ["BAKER~ Sydney_Clipped.edf", "_"],
        ["BRADFORD~ Charles_Clipped.edf", "_"],
        ["BRAUN~ Brooke_Clipped.edf", "_"],
        ["BROWN~ Kaidience_Clipped.edf", "_"],
        ["BUCCAT~ Angelica_Clipped.edf", "_"],
        ["COSTALL~ Logan_Clipped.edf", "_"],
        ["DAY~ James_Clipped.edf", "_"],
        ["DERBYSHIRE~ Natalie_Clipped.edf", "_"],
        ["DINGHA~ Chabod_Clipped.edf", "_"],
        ["DZAWANDA~ Tanatsiwa_Clipped.edf", "_"],
        ["GAGNON~ Rebecca_Clipped.edf", "_"],
        ["GAMMON~ Ciara_Clipped.edf", "_"],
        ["GAUTHIER~ Jocelyn_Clipped.edf", "_"],
        ["HARLEY-STUCKEY_Brooklynn_Clipped.edf", "_"],
        ["HOCKLEY~ Olivia_Clipped.edf", "_"],
        ["HURRY~ Conner_Clipped.edf", "_"],
        ["IDRIS~ IMAN_Clipped.edf", "_"],
        ["IRWIN~ Benjamin_Clipped.edf", "_"],
        ["LEMBOYE~ Anne_Clipped.edf", "_"],
        ["LUBON~ Gabriela_Clipped.edf", "_"],
        ["LYNCH~ Marnie_Clipped.edf", "_"],
        ["LYNNIK~ Iuliia_Clipped.edf", "_"],
        ["MOFFAT~ Joshua_Clipped.edf", "_"],
        ["MOSQUITO~ Raven_Clipped.edf", "_"],
        ["NYAR~ NawWah_Clippped.edf", "_"],
        ["PROUD~ Brooke-lynne_Clipped.edf", "_"],
        ["QUELCH~ Theo_Clipped.edf", "_"],
        ["REIS~ Sarah_Clipped.edf", "_"],
        ["SIERRA~ Bianca_Clipped.edf", "_"],
        ["SLATER-LOTTER~Petra_Clipped_.edf", "_"],
        ["STOESSER~ Rhett_Clipped.edf", "_"],
        ["SUMMERS~ Karyssa_Clipped.edf", "_"],
        ["THOMAS~ David_Clipped.edf", "_"],
        ["TURNER~ Ethan_Clipped.edf", "_"],
        ["VAUGHN~ Robert_Clipped.edf", "_"],
        ["VOSBURGH~ Ashley_Clipped.edf", "_"],
        ["WAGONER~ Kenneth_Clipped.edf", "_"],
        ["WALKER~ Marguerite_Clipped.edf", "_"],
        ["WATSON~ Kindigo_Clipped.edf", "_"],
        ["WENGER~ Portia_Clipped.edf", "_"],
        ["WEST~ Dustin_Clipped.edf", "_"],
        ["WHITE~ Clare_Clipped.edf", "_"],
        ["WHYBURD~ Rita_Clipped.edf", "_"],
        ["ZILKA~ Rory_Clipped.edf", "_"],
        ["BENMOSHE~ Yakov_Clipped.edf", ""],
        ["CARROLL~ Lennon_Clipped.edf", ""],
        ["HADEN~ Joe_Clipped.edf", ""],
        ["KEIM~ Zachary_Clipped.edf", ""],
        ["MONTENEGRORAMI_Daniel_Clipped.edf", ""],
        ["MUHAMMAD~ Rafeh_Clipped.edf", ""],
        ["SANDERS~ Anton_Clipped.edf", ""],
        ["ARNTZEN~ Matthew_Clipped.edf", ""],
        ["CAMPBELL~ Finnen_Clipped.edf", ""],
        ]
        
        return dataset_name, data_path, eeg_files_info   

    def Persyst_HFO_EPILEPSIAE(self):
        dataset_name = "Persyst_HFO_EPILEPSIAE"
        data_path = Path("F:/FREIBURG_Simultaneous_OneHrFiles/")

        pats_ls = [
            "FR_1073",
            "FR_1084",
            "FR_1096",
            "FR_1125",
            "FR_442",
            "FR_548",
            "FR_590",
            "FR_916",
        ]

        nr_files_include = 52

        files_dict = {'PatName': [], 'Filepath': []}
        for path in data_path.glob("**/*.lay"):
            fname = path.parts[-1]
            pat_name = path.parts[-2].replace('pat_', '')
            if pat_name in pats_ls:
                files_dict['PatName'].append(pat_name)
                files_dict['Filepath'].append(path)

        files_dict_trim = {'PatName': [], 'Filepath': []}
        pat_names = list(set(files_dict['PatName']))
        pat_names.sort()
        for pat_name in pat_names:
            pat_fpaths = [fpath for p_name, fpath in zip( files_dict['PatName'], files_dict['Filepath']) if p_name == pat_name]
            pat_fpaths = pat_fpaths[0:nr_files_include]
            pat_name_ls = [pat_name]*len(pat_fpaths)
            files_dict_trim['PatName'].extend(pat_name_ls)
            files_dict_trim['Filepath'].extend(pat_fpaths)

        return dataset_name, files_dict_trim

    def Multidetect_27_SOZ(self):
        dataset_name = "PersystMultidetect_27_SOZ_HFO_EPILEPSIAE"
        data_path = Path("C:/Users/HFO/Documents/Postdoc_Calgary/Research/Spectral_HFO_SOZ_Prediction/ACH/EEG_Data")        

        files_dict = {'PatName': [], 'Filepath': []}
        for path in data_path.glob("**/*.vhdr"):
            fname = path.parts[-1]
            files_dict['PatName'].append(fname)
            files_dict['Filepath'].append(path)

        return dataset_name, files_dict
    
    def ACH_27_Multidetect_SOZ_Study(self):
        dataset_name = "ACH_27_Multidetect_SOZ_Study"       
        data_path = ""
        
        data_path = "/work/jacobs_lab/ACH/"
        
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = ""
        elif socket.gethostname() == "DLP":
            data_path = ""
        elif socket.gethostname() == "dlp":
            data_path = "/media/dlp/SpikeDrive/Spectral_HFO_SOZ_Prediction/MNI/EEG_Data/"


        data_path = Path(data_path)
        files_dict = {'PatName': [], 'Filepath': []}
        for path in data_path.glob("**/*.vhdr"):
            fname = path.parts[-1]
            files_dict['PatName'].append(fname)
            files_dict['Filepath'].append(path)

        return dataset_name, files_dict
    
    def Anonymized_Physio_Patients_Study(self):
        
        group="HFOHealthy1monto2yrs"
        # group="HFOHealthy3to5yrs"
        # group="HFOHealthy6to10yrs"
        # group="HFOHealthy11to13yrs"
        # group="HFOHealthy14to17yrs"

        dataset_name = f"Anonymized_Physio_Patients_Study/{group}"       
        data_path = ""
        
        data_path = f"/work/jacobs_lab/PhysioEEG/{group}/"
        
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = ""
        elif socket.gethostname() == "DLP":
            data_path = ""
        elif socket.gethostname() == "dlp":
            data_path = "/media/dlp/SpikeDrive/Spectral_HFO_SOZ_Prediction/MNI/EEG_Data/"


        data_path = Path(data_path)
        files_dict = {'PatName': [], 'Filepath': []}
        for path in data_path.glob("**/*.edf"):
            fname = path.parts[-1]
            files_dict['PatName'].append(fname)
            files_dict['Filepath'].append(path)

        return dataset_name, files_dict

if __name__ == "__main__":
    StudiesInfo().Multidetect_27_SOZ()

    pass
