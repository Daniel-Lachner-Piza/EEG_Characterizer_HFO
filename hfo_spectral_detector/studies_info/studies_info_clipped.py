
import os
import socket

class StudiesInfo:
    def __init__(self) -> None:
        pass
    
    def Maggi_Stroke_Data_1_clipped(self):
        dataset_name = "Maggi_Stroke_Data_1"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_Data_1/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_Data_1/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BAI~ Male_18852025-4e7c-4771-b29c-1680b3c248b6_3.2_annotated_clip.edf", "BAI~ Male_18852025-4e7c-4771-b29c-1680b3c248b6_3.2_annotations_clipped_eeg.mat"],
        ["BAI~ Male_488594c9-ce54-44fe-87cd-e7e737ac61a1_clip2corr_annotated_clip.edf", "BAI~ Male_488594c9-ce54-44fe-87cd-e7e737ac61a1_clip2corr_annotations_clipped_eeg.mat"],
        ["BAI~ Male_58bc137c-0b47-4d92-bbd7-62755aba5b8b_clip1.1_annotated_clip.edf", "BAI~ Male_58bc137c-0b47-4d92-bbd7-62755aba5b8b_clip1.1_annotations_clipped_eeg.mat"],
        ["BAI~ Male_d65a5a12-3139-4d9f-ba62-85692393b768_clip3.1_annotated_clip.edf", "BAI~ Male_d65a5a12-3139-4d9f-ba62-85692393b768_clip3.1_annotations_clipped_eeg.mat"],
        ["DOLAN~ Matteo_338baf73-b835-4d5a-844d-230fd11b6f9d_clip2longcheckSNR_annotated_clip.edf", "DOLAN~ Matteo_338baf73-b835-4d5a-844d-230fd11b6f9d_clip2longcheckSNR_annotations_clipped_eeg.mat"],
        ["DOLAN~ Matteo_ac6ef925-7cdd-45e7-9df0-4ec1f61d5dfa_clip3_annotated_clip.edf", "DOLAN~ Matteo_ac6ef925-7cdd-45e7-9df0-4ec1f61d5dfa_clip3_annotations_clipped_eeg.mat"],
        ["FLORESLEDO~ Fe_4f59ff22-bf9a-42bd-afe2-81a75afbbce1_clip3.2_annotated_clip.edf", "FLORESLEDO~ Fe_4f59ff22-bf9a-42bd-afe2-81a75afbbce1_clip3.2_annotations_clipped_eeg.mat"],
        ["FLORESLEDO~ Fe_77328de8-32a8-40ba-befd-cc28e21e169d_clip2_annotated_clip.edf", "FLORESLEDO~ Fe_77328de8-32a8-40ba-befd-cc28e21e169d_clip2_annotations_clipped_eeg.mat"],
        ["FLORESLEDO~ Fe_fd4fbe0a-d55a-4607-be05-0977ed1f09de_clip1extra short_annotated_clip.edf", "FLORESLEDO~ Fe_fd4fbe0a-d55a-4607-be05-0977ed1f09de_clip1extra short_annotations_clipped_eeg.mat"],
        ["GERMERSHAUSEN~_b9c91e38-8fe4-4f8d-ad84-7be9f91d2773_clip1_annotated_clip.edf", "GERMERSHAUSEN~_b9c91e38-8fe4-4f8d-ad84-7be9f91d2773_clip1_annotations_clipped_eeg.mat"],
        ["KAUR~ Male_425002f3-2232-440f-8a2f-900fadf664c3_clip2_annotated_clip.edf", "KAUR~ Male_425002f3-2232-440f-8a2f-900fadf664c3_clip2_annotations_clipped_eeg.mat"],
        ["KAUR~ Male_bb3050be-5569-48ef-853d-858dc766e0a5_clip3_annotated_clip.edf", "KAUR~ Male_bb3050be-5569-48ef-853d-858dc766e0a5_clip3_annotations_clipped_eeg.mat"],
        ["KAUR~ Male_ef26672f-9961-4b85-b961-2d125f70f69c_clip1_annotated_clip.edf", "KAUR~ Male_ef26672f-9961-4b85-b961-2d125f70f69c_clip1_annotations_clipped_eeg.mat"],
        ["MAGNUSON~ Fema_8ba6e632-3588-4051-a449-98bd621015bb_clip1_annotated_clip.edf", "MAGNUSON~ Fema_8ba6e632-3588-4051-a449-98bd621015bb_clip1_annotations_clipped_eeg.mat"],
        ["MAGNUSON~ Fema_c0be2f31-7282-49ba-b19d-da14e250f064_clip2_annotated_clip.edf", "MAGNUSON~ Fema_c0be2f31-7282-49ba-b19d-da14e250f064_clip2_annotations_clipped_eeg.mat"],
        ["MCLEAN~ Male_319eb166-fdef-4b49-8738-0ec01bb259f4_clip1_annotated_clip.edf", "MCLEAN~ Male_319eb166-fdef-4b49-8738-0ec01bb259f4_clip1_annotations_clipped_eeg.mat"],
        ["MCLEAN~ Male_3f1d31f8-c2d4-41fc-8fdb-134c22ca2838_clip3alternative_annotated_clip.edf", "MCLEAN~ Male_3f1d31f8-c2d4-41fc-8fdb-134c22ca2838_clip3alternative_annotations_clipped_eeg.mat"],
        ["MCLEAN~ Male_559c2d9b-0af5-4dcb-8e3c-2c9515ac0b06_clip2_annotated_clip.edf", "MCLEAN~ Male_559c2d9b-0af5-4dcb-8e3c-2c9515ac0b06_clip2_annotations_clipped_eeg.mat"],
        ["MCLEAN~ Male_de41c5b4-4c44-4872-877f-5b7b73a2e2ab_clip3_annotated_clip.edf", "MCLEAN~ Male_de41c5b4-4c44-4872-877f-5b7b73a2e2ab_clip3_annotations_clipped_eeg.mat"],
        ["McNAUGHTON~ Ma_60fd4778-8fca-4e65-89c1-3599af0e0782_clip1_annotated_clip.edf", "McNAUGHTON~ Ma_60fd4778-8fca-4e65-89c1-3599af0e0782_clip1_annotations_clipped_eeg.mat"],
        ["McNAUGHTON~ Ma_643439e4-8bfe-41fd-800f-63cc792679d0_clip4_annotated_clip.edf", "McNAUGHTON~ Ma_643439e4-8bfe-41fd-800f-63cc792679d0_clip4_annotations_clipped_eeg.mat"],
        ["McNAUGHTON~ Ma_e631d7db-ca9c-4a09-b6bc-cdaa4e96960e_clip3_annotated_clip.edf", "McNAUGHTON~ Ma_e631d7db-ca9c-4a09-b6bc-cdaa4e96960e_clip3_annotations_clipped_eeg.mat"],
        ["MEGA~ Male_635ef0ab-5eb3-42cc-a2a9-866e6fca5f65_clip1new_annotated_clip.edf", "MEGA~ Male_635ef0ab-5eb3-42cc-a2a9-866e6fca5f65_clip1new_annotations_clipped_eeg.mat"],
        ["MEGA~ Male_73aff7f3-ed20-42c7-9f52-5b4b1c2ae931_clip3_annotated_clip.edf", "MEGA~ Male_73aff7f3-ed20-42c7-9f52-5b4b1c2ae931_clip3_annotations_corrected.mat"],
        ["MEGA~ Male_a3bbfc58-76bf-41c9-af8c-4061a4cd206d_clip2_annotated_clip.edf", "MEGA~ Male_a3bbfc58-76bf-41c9-af8c-4061a4cd206d_clip2_annotations_clipped_eeg.mat"],
        ["MEGA~ Male_dd9c7e99-9feb-451a-ae62-acc110ba5153_clip4_annotated_clip.edf", "MEGA~ Male_dd9c7e99-9feb-451a-ae62-acc110ba5153_clip4_annotations_clipped_eeg.mat"],
        ["NEILSONTRUMBLE_222d821c-0540-4892-8d72-34894bf93754_clip1_annotated_clip.edf", "NEILSONTRUMBLE_222d821c-0540-4892-8d72-34894bf93754_clip1_annotations_clipped_eeg.mat"],
        ["NEILSONTRUMBLE_2dc3d459-cea6-409e-8f73-ee2704f9cdfd_clip2_annotated_clip.edf", "NEILSONTRUMBLE_2dc3d459-cea6-409e-8f73-ee2704f9cdfd_clip2_annotations_clipped_eeg.mat"],
        ["PRICE~ Weston_01073139-8028-4dd3-a867-199467e470d4_clip1corr1_annotated_clip.edf", "PRICE~ Weston_01073139-8028-4dd3-a867-199467e470d4_clip1corr1_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_25854595-03b8-42d4-b1d3-0c1991d03911_clip5_annotated_clip.edf", "QADIR~ Fathiya_25854595-03b8-42d4-b1d3-0c1991d03911_clip5_2annotations.mat"],
        ["QADIR~ Fathiya_307b2e7f-7fd0-4c74-b4a1-97721abcfd79_clip1.2_annotated_clip.edf", "QADIR~ Fathiya_307b2e7f-7fd0-4c74-b4a1-97721abcfd79_clip1.2_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_60fe42da-518b-46e8-8cd3-a70b499f6a23_clip4_annotated_clip.edf", "QADIR~ Fathiya_60fe42da-518b-46e8-8cd3-a70b499f6a23_clip4_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_a1f023d8-ce00-4d4e-a379-8f98baa145f3_clip1_annotated_clip.edf", "QADIR~ Fathiya_a1f023d8-ce00-4d4e-a379-8f98baa145f3_clip1_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_b3c6d9ca-c1c2-46f9-ae42-4b7e68ac0568_clip4corrnew_annotated_clip.edf", "QADIR~ Fathiya_b3c6d9ca-c1c2-46f9-ae42-4b7e68ac0568_clip4corrnew_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_c7d184ce-56a5-42c4-8c7c-a26186d60e42_clip2(long,better)_annotated_clip.edf", "QADIR~ Fathiya_c7d184ce-56a5-42c4-8c7c-a26186d60e42_clip2(long,better)_annotations_corrected.mat"],
        ["QADIR~ Fathiya_cfea5f8e-eb97-471e-a1d7-790f14684489_clip6_annotated_clip.edf", "QADIR~ Fathiya_cfea5f8e-eb97-471e-a1d7-790f14684489_clip6_annotations_clipped_eeg.mat"],
        ["QADIR~ Fathiya_dfd6d595-3a4c-4f8c-b4af-0d196b1eb981_clip3.1_annotated_clip.edf", "QADIR~ Fathiya_dfd6d595-3a4c-4f8c-b4af-0d196b1eb981_clip3.1_annotations_clipped_eeg.mat"],
        ["SPROTT~ Male_97836b6d-ccf0-4f8d-a22e-670701771796_clip1_annotated_clip.edf", "SPROTT~ Male_97836b6d-ccf0-4f8d-a22e-670701771796_clip1_annotations_clipped_eeg.mat"],
        ["SPROTT~ Male_f7bc4d42-3804-4cdd-94ae-a126c2fa02b8_clip2_annotated_clip.edf", "SPROTT~ Male_f7bc4d42-3804-4cdd-94ae-a126c2fa02b8_clip2_annotations_clipped_eeg.mat"],


        ]

        return dataset_name, data_path, eeg_files_info
    
    def Maggi_Stroke_Data_2_clipped(self):
        dataset_name = "Maggi_Stroke_Data_2"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_Data_2_Jan2024_ForRome/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_Data_2_Jan2024_ForRome/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["AMBROSIO~ Kyle_68375d1d-96c5-43cd-a5fa-c38547650e78_annotated_clip.edf", "AMBROSIO~ Kyle_68375d1d-96c5-43cd-a5fa-c38547650e78_annotations_clipped_eeg.mat"],
        ["BAHTA~ Dawit_2eea09fa-d77d-4acc-8012-4aae693d48ad_annotated_clip.edf", "BAHTA~ Dawit_2eea09fa-d77d-4acc-8012-4aae693d48ad_annotations_clipped_eeg.mat"],
        ["BHATTI~ Arjun_014d5a1a-0532-44f1-90a5-b05dce073131_annotated_clip.edf", "BHATTI~ Arjun_014d5a1a-0532-44f1-90a5-b05dce073131_annotations_clipped_eeg.mat"],
        ["DOYLE~ Hannah_efbb65d1-0ac3-4e24-bfdd-6e9f35c6dc3e_annotated_clip.edf", "DOYLE~ Hannah_efbb65d1-0ac3-4e24-bfdd-6e9f35c6dc3e_annotations_clipped_eeg.mat"],
        ["ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotated_clip.edf", "ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotations_clipped_eeg.mat"],
        ["GAREAU~ Braede_1007dcde-cd78-447d-99f7-823d82eec545_LTM1norm_annotated_clip.edf", "GAREAU~ Braede_1007dcde-cd78-447d-99f7-823d82eec545_LTM1norm_annotations_clipped_eeg.mat"],
        ["GOBLELIDSTONE~_d43e27d4-4076-4150-a0d5-53eca03db13c_annotated_clip.edf", "GOBLELIDSTONE~_d43e27d4-4076-4150-a0d5-53eca03db13c_annotations_clipped_eeg.mat"],
        ["HAMMADI~ Ayoub_bc1a1adb-d804-4bac-a07a-d0628b13d89a_annotated_clip.edf", "HAMMADI~ Ayoub_bc1a1adb-d804-4bac-a07a-d0628b13d89a_annotations_clipped_eeg.mat"],
        ["MASAUD~ Mohamm_cc4de571-ea29-4a8f-b4b2-14047db7eb5e_annotated_clip.edf", "MASAUD~ Mohamm_cc4de571-ea29-4a8f-b4b2-14047db7eb5e_annotations_clipped_eeg.mat"],
        ["McMILLAN~ Sydn_6dfb44c7-36ce-48ad-90cc-d967ffb824dc_annotated_clip.edf", "McMILLAN~ Sydn_6dfb44c7-36ce-48ad-90cc-d967ffb824dc_annotations_clipped_eeg.mat"],
        ["MILLS~ Abigail_cebcb472-3af5-42d3-b688-efb49c01e58f_annotated_clip.edf", "MILLS~ Abigail_cebcb472-3af5-42d3-b688-efb49c01e58f_annotations_clipped_eeg.mat"],
        ["PEZDERIC~ Kier_2333d768-06a9-4730-9900-dcd3381df60d_annotated_clip.edf", "PEZDERIC~ Kier_2333d768-06a9-4730-9900-dcd3381df60d_annotations_clipped_eeg.mat"],
        ["PRICE~ Weston_7bd004d0-e2b6-42d3-b1b3-5126788b7918_annotated_clip.edf", "PRICE~ Weston_7bd004d0-e2b6-42d3-b1b3-5126788b7918_annotations_clipped_eeg.mat"],
        ["SINJAR~ Majd_286a1fc0-7679-4785-81d7-2a0f1ec1a11e_annotated_clip.edf", "SINJAR~ Majd_286a1fc0-7679-4785-81d7-2a0f1ec1a11e_annotations_clipped_eeg.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    
    def Maggi_Stroke_Data_3_clipped(self):
        dataset_name = "Maggi_Stroke_Data_3"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_AES2024_2ndPart/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Maggi_Stroke_AES2024_2ndPart/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["ALI~ Ume_00cb8b62-12e2-4f78-8170-79f64fb15079_annotated_clip.edf","ALI~ Ume_00cb8b62-12e2-4f78-8170-79f64fb15079_24_6_2024_annotations_clipped_eeg.mat"],
        ["AMBROSIO~ Kyle_cat2!!_annotated_clip.edf","AMBROSIO~ Kyle_cat2!!_annotations_clipped_eeg.mat"],
        ["AUBE~ Sophia_5936864d-e163-4a56-afd5-45fe1003e138_annotated_clip.edf","AUBE~ Sophia_5936864d-e163-4a56-afd5-45fe1003e138_annotations_clipped_eeg.mat"],
        ["BHATTI~ Arjun_better quality_annotated_clip.edf","BHATTI~ Arjun_better quality_2_annotations_clipped_eeg.mat"],
        ["DHILLON~ Gurma_9e9c553e-46c9-4733-99c6-f2bb30dd95ba_annotated_clip.edf","DHILLON~ Gurma_annotations_clipped_eeg.mat"],
        ["ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotated_clip.edf","ECKSTRAND~ Car_cc1ecbfe-70fe-4b45-b7bc-70b0451f86a6_annotations_clipped_eeg.mat"],
        ["GATES~ Rylan_ecead51a-1352-4f42-bbdb-af0d041f3372_annotated_clip.edf","GATES~ Rylan_ecead51a-1352-4f42-bbdb-af0d041f3372_annotations_clipped_eeg.mat"],
        ["HARTNETT~ Juli_558d4ced-f4ad-44f5-92db-c6364d7d9a1d_annotated_clip.edf","HARTNETT~ Juli_558d4ced-f4ad-44f5-92db-c6364d7d9a1d_annotations_clipped_eeg.mat"],
        ["HILDEBRANDT~ C_489e2278-08f6-4b6c-ab50-65a76042c714_annotated_clip.edf","HILDEBRANDT~ C_489e2278-08f6-4b6c-ab50-65a76042c714_annotations_clipped_eeg.mat"],
        ["LUNN~ Madison_cat2_annotated_clip.edf","LUNN~ Madison_cat2_annotations_clipped_eeg.mat"],
        ["MUSCEDERE~ Sor_6d48a910-f98a-44e6-bbf7-dbbe5a8e0252_annotated_clip.edf","MUSCEDERE~ Sor_6d48a910-f98a-44e6-bbf7-dbbe5a8e0252_annotations_clipped_eeg.mat"],
        ["NSOUM~ Denise_cd854db2-a82a-4aa8-9b36-aeedd42b3e29_annotated_clip.edf","NSOUM~ Denise_cd854db2-a82a-4aa8-9b36-aeedd42b3e29_annotations_clipped_eeg.mat"],
        ["PARRAPENA~ Dan_b3944205-b98d-43c1-a6ad-09c1cb8462a1_annotated_clip.edf","PARRAPENA~ Dan_b3944205-b98d-43c1-a6ad-09c1cb8462a1_annotations_clipped_eeg.mat"],
        ["WHITTY~ Rachel_72648762-a94d-4088-9bb1-3608b753851c_annotated_clip.edf","WHITTY~ Rachel_72648762-a94d-4088-9bb1-3608b753851c_annotations_clipped_eeg.mat"],
        ]

        return dataset_name, data_path, eeg_files_info

    def Minette_AED_clipped(self):
        dataset_name = "Minette_AED"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Minette_AED/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Minette_AED/"
            
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BENNETT~ Nicol_e7d45741-4045-4aa8-983a-205275eb1d7bmar2023_annotated_clip.edf", "BENNETT~ Nicol_e7d45741-4045-4aa8-983a-205275eb1d7bmar2023_24_7_2023__2_57_4_annotations_clipped_eeg.mat"],
        ["BENNETT~ Nicol_ea448b98-6869-4e7a-86bd-14a44952098daug2021_annotated_clip.edf", "BENNETT~ Nicol_ea448b98-6869-4e7a-86bd-14a44952098daug2021_1_8_2023__12_22_21_annotations_clipped_eeg.mat"],
        ["FRASER~ Nealen_899657ff-f3ff-41eb-8561-4eb4e80ee6b5mar2023_annotated_clip.edf", "FRASER~ Nealen_899657ff-f3ff-41eb-8561-4eb4e80ee6b5mar2023_8_8_2023__16_53_21_annotations_clipped_eeg.mat"],
        ["FRASER~ Nealen_8f7bb0c0-26be-40b8-b6e2-8a4de5aec013feb2022_annotated_clip.edf", "FRASER~ Nealen_8f7bb0c0-26be-40b8-b6e2-8a4de5aec013feb2022_22_9_2023__9_18_57_annotations_clipped_eeg.mat"],
        ["FRIESEN~ Dawso_99b634bc-ce27-4d83-936a-7e6f9c8407cb_annotated_clip.edf", "FRIESEN~ Dawso_99b634bc-ce27-4d83-936a-7e6f9c8407cb_18_9_2023__15_19_29_annotations_clipped_eeg.mat"],
        ["FRIESEN~ Dawso_de7d355d-47f9-4a2f-a97f-d5287e615c8f_annotated_clip.edf", "FRIESEN~ Dawso_de7d355d-47f9-4a2f-a97f-d5287e615c8f_18_9_2023__18_10_10_annotations_clipped_eeg.mat"],
        ["GAETZ_ Phoebe_40bab3e7-b02f-4704-ae3e-b67a5af3130c_annotated_clip.edf", "GAETZ_ Phoebe_40bab3e7-b02f-4704-ae3e-b67a5af3130c_21_7_2023__5_59_35_annotations_clipped_eeg.mat"],
        ["GAETZ~ Phoebe_ca90e0b3-3763-41c1-9a52-308618efd77e_annotated_clip.edf", "GAETZ~ Phoebe_ca90e0b3-3763-41c1-9a52-308618efd77e_27_5_2023__16_1_29_annotations_clipped_eeg.mat"],
        ["HONEYWOOD_ Cha_0c5008f8-5d1d-43d5-89cf-f815f75aae97_annotated_clip.edf", "HONEYWOOD_ Cha_0c5008f8-5d1d-43d5-89cf-f815f75aae97_19_9_2023__15_56_9_annotations_clipped_eeg.mat"],
        ["HONEYWOOD_ Cha_e3de88cc-84cc-40ae-ba45-504bfb148760_annotated_clip.edf", "HONEYWOOD_ Cha_e3de88cc-84cc-40ae-ba45-504bfb148760_22_9_2023__8_49_29_annotations_clipped_eeg.mat"],
        ["LONGARD~ Callu_1e8e25e0-8f2d-4de1-8422-d6cf88547fb8june2022_annotated_clip.edf", "LONGARD~ Callu_1e8e25e0-8f2d-4de1-8422-d6cf88547fb8june2022_10_8_2023__12_38_51_annotations_clipped_eeg.mat"],
        ["LONGARD~ Callu_ce820813-d31c-4afe-9395-2d424bb58444feb2023_annotated_clip.edf", "LONGARD~ Callu_ce820813-d31c-4afe-9395-2d424bb58444feb2023_13_8_2023__15_21_3_annotations_clipped_eeg.mat"],
        ["MEIER~ Jenna_43dd9467-aa86-4599-916c-4c43448ca19b_annotated_clip.edf", "MEIER~ Jenna_43dd9467-aa86-4599-916c-4c43448ca19b_7_5_2023__18_48_7_annotations_clipped_eeg.mat"],
        ["MEIER~ Jenna_872fb9c8-cce6-4ebb-88f8-5dc473fa320a_annotated_clip.edf", "MEIER~ Jenna_872fb9c8-cce6-4ebb-88f8-5dc473fa320a_4_5_2023__17_50_7_annotations_clipped_eeg.mat"],
        ["MOFFET~ Cooper_38fbd417-2f6f-48fb-9b2c-55ace040e6cb_annotated_clip.edf", "MOFFET~ Cooper_38fbd417-2f6f-48fb-9b2c-55ace040e6cb_28_4_2023__17_39_18_annotations_clipped_eeg.mat"],
        ["MOFFET~ Cooper_e8b864b9-8856-4195-b0d7-6c38040534c7_annotated_clip.edf", "MOFFET~ Cooper_e8b864b9-8856-4195-b0d7-6c38040534c7_18_4_2023__12_27_52_annotations_clipped_eeg.mat"],
        ["MONAGHAN~ Oliv_08d952de-7597-48a1-a44a-6e7a1c9cd219May2023_annotated_clip.edf", "MONAGHAN~ Oliv_08d952de-7597-48a1-a44a-6e7a1c9cd219May2023_14_8_2023__10_38_41_annotations_clipped_eeg.mat"],
        ["MONAGHAN~ Oliv_dba22cb3-1a13-488e-b901-9eb508919918oct2021_annotated_clip.edf", "MONAGHAN~ Oliv_dba22cb3-1a13-488e-b901-9eb508919918oct2021_14_8_2023__16_12_36_annotations_clipped_eeg.mat"],
        ["OBOYLE~ Holly_e013d0bf-e2c0-4782-8733-fb34dd0a1984april2022_annotated_clip.edf", "OBOYLE~ Holly_e013d0bf-e2c0-4782-8733-fb34dd0a1984april2022_17_8_2023__11_30_3_annotations_clipped_eeg.mat"],
        ["OBOYLE~ Holly_e42b10a8-c4db-41f9-9f40-a3d4fe788714May 3, 2023_annotated_clip.edf", "OBOYLE~ Holly_e42b10a8-c4db-41f9-9f40-a3d4fe788714May 3, 2023_20_8_2023__11_53_19_annotations_clipped_eeg.mat"],
        ["OKEKE~ Oge_92a42279-6045-43d6-a2e0-c61c66463b04apr2021_annotated_clip.edf", "OKEKE~ Oge_92a42279-6045-43d6-a2e0-c61c66463b04apr2021_24_8_2023__15_27_3_annotations_clipped_eeg.mat"],
        ["OKEKE~ Oge_ab74fbce-5f09-4a84-bc90-f1de604549ecjune2023_annotated_clip.edf", "OKEKE~ Oge_ab74fbce-5f09-4a84-bc90-f1de604549ecjune2023(1)_31_8_2023__14_27_24_annotations_clipped_eeg.mat"],
        ["PALLARESPULIDO_9d9e9b46-a26f-46fe-b360-39dfca75c372may2023_annotated_clip.edf", "PALLARESPULIDO_9d9e9b46-a26f-46fe-b360-39dfca75c372may2023_29_8_2023__10_47_34_annotations_clipped_eeg.mat"],
        ["PALLARESPULIDO_f6f10281-d5f0-4e05-afd5-78db01268666nov2021_annotated_clip.edf", "PALLARESPULIDO_f6f10281-d5f0-4e05-afd5-78db01268666nov2021_30_8_2023__14_41_34_annotations_clipped_eeg.mat"],
        ["POWELL~ Morgan_51e4ebea-fd43-4884-a8eb-48f1bcf4d2a5april2023_annotated_clip.edf", "POWELL~ Morgan_51e4ebea-fd43-4884-a8eb-48f1bcf4d2a5april2023_5_9_2023__10_18_3_annotations_clipped_eeg.mat"],
        ["POWELL~ Morgan_66c348de-93aa-43a9-b91a-9c8a978a4f5fsept2022_annotated_clip.edf", "POWELL~ Morgan_66c348de-93aa-43a9-b91a-9c8a978a4f5fsept2022_7_9_2023__13_2_8_annotations_clipped_eeg.mat"],
        ["QUINTON~ Morga_23e9f290-1010-4669-a90f-87c458daffad_annotated_clip.edf", "QUINTON~ Morga_23e9f290-1010-4669-a90f-87c458daffad_3_4_2023__13_34_3_annotations_clipped_eeg.mat"],
        ["QUINTON~ Morga_651e748b-616f-4741-858a-75955aae8968_annotated_clip.edf", "QUINTON~ Morga_651e748b-616f-4741-858a-75955aae8968_31_3_2023__15_24_16_annotations_clipped_eeg.mat"],
        ["RITSON-BENNET~_180f1128-850b-4769-a9fe-4145d0b1a3ea_annotated_clip.edf", "RITSON-BENNET~_180f1128-850b-4769-a9fe-4145d0b1a3ea_31_3_2023__9_53_4_annotations_clipped_eeg.mat"],
        ["RITSON-BENNETT_022648e5-617e-46fc-b974-22f9c441808f_annotated_clip.edf", "RITSON-BENNETT_022648e5-617e-46fc-b974-22f9c441808f_29_3_2023__16_48_33_annotations_clipped_eeg.mat"],
        ["SABADO_ Marek_af6286cd-0f6a-4055-b69f-1f32bc66477b_annotated_clip.edf", "SABADO_ Marek_af6286cd-0f6a-4055-b69f-1f32bc66477b (1)_6_3_2023__14_56_35_annotations_clipped_eeg.mat"],
        ["SABADO~ Marek_df512d13-ca61-4c54-af6a-0f25b25fd8fe_annotated_clip.edf", "SABADO~ Marek_df512d13-ca61-4c54-af6a-0f25b25fd8fe_20_3_2023__8_41_50_annotations_clipped_eeg.mat"],
        ["STOCKMAN~ TYLE_3ed33d0e-ba4a-4969-b586-86fa9b551131_annotated_clip.edf", "STOCKMAN~ TYLE_3ed33d0e-ba4a-4969-b586-86fa9b551131_3_3_2023__18_11_59_annotations_clipped_eeg.mat"],
        ["STOCKMAN~ TYLE_b279967e-2ef3-425a-8d7e-03642c7fc042_annotated_clip.edf", "STOCKMAN~ TYLE_b279967e-2ef3-425a-8d7e-03642c7fc042_13_2_2023__11_45_35_annotations_clipped_eeg.mat"],
        ["West_Parker-mar-2023_3059749451_annotated_clip.edf", "West Parker-mar-2023_3059749451_11_9_2023__11_59_32_annotations_clipped_eeg.mat"],
        ["WEST~ Parker_9ee4fbb9-0196-47a1-91ac-ba707f346447April2022_annotated_clip.edf", "WEST~ Parker_9ee4fbb9-0196-47a1-91ac-ba707f346447April2022_14_9_2023__16_23_39_annotations_clipped_eeg.mat"],
        ["WILDEMAN~ Taiv_45c8bf5e-1200-42bb-bfb8-c4f08c6efa7c_annotated_clip.edf", "WILDEMAN~ Taiv_45c8bf5e-1200-42bb-bfb8-c4f08c6efa7c_20_9_2023__23_2_43_annotations_clipped_eeg.mat"],
        ["WILDEMAN~ Taiv_e8db78f1-17c7-4820-ac0f-7562bfe19a97_annotated_clip.edf", "WILDEMAN~ Taiv_e8db78f1-17c7-4820-ac0f-7562bfe19a97_21_9_2023__17_37_10_annotations_clipped_eeg.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    def Multidetect_Validated_HFO_clipped(self):
        dataset_name = "Multidetect_Validated_HFO"
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Multidetect_Maggi_Validated_Zurich_Pats/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Scalp_HFO_GoldStandard/Annotated_EEG_Clips/Multidetect_Maggi_Validated_Zurich_Pats/"
        # eeg file, annotation file
        eeg_files_info = \
        [
        ["BONERTZ~ Denve_f651bec5-8213-4df8-a500-e5cde80eed6d_annotated_clip.vhdr", "BONERTZ~ Denve_f651bec5-8213-4df8-a500-e5cde80eed6d_annotations_clipped_eeg.mat"],
        ["BRAVO~ Isabell_dcda0647-7aba-49f7-a403-6677416e3e93_annotated_clip.vhdr", "BRAVO~ Isabell_dcda0647-7aba-49f7-a403-6677416e3e93_annotations_clipped_eeg.mat"],
        ["BRIGGSRIGIO~ S_f4d226da-a40f-4ab6-91fa-cf8290227e22_annotated_clip.vhdr", "BRIGGSRIGIO~ S_f4d226da-a40f-4ab6-91fa-cf8290227e22_annotations_clipped_eeg.mat"],
        ["CONTRERAS~ Hol_b747da66-1ae0-4f27-bbf8-ce6e92160a52_annotated_clip.vhdr", "CONTRERAS~ Hol_b747da66-1ae0-4f27-bbf8-ce6e92160a52_annotations_clipped_eeg.mat"],
        ["DEDRICK~ Olivi_412b3f13-ebc6-4ed5-bc2d-792450c72e84_annotated_clip.vhdr", "DEDRICK~ Olivi_412b3f13-ebc6-4ed5-bc2d-792450c72e84_annotations_clipped_eeg.mat"],
        ["DEVER~ Rory_db8f3b9b-90cf-4ad7-a253-2f28d798fc05_annotated_clip.vhdr", "DEVER~ Rory_db8f3b9b-90cf-4ad7-a253-2f28d798fc05_annotations_clipped_eeg.mat"],
        ["FREI~ Domenica_9c81500d-872f-42c6-a0ee-d6d8f4c23e2f_annotated_clip.vhdr", "FREI~ Domenica_9c81500d-872f-42c6-a0ee-d6d8f4c23e2f_annotations_clipped_eeg.mat"],
        ["GOBLELIDSTONE~_1f68f041-84ac-41b0-be3d-638d17500c7a_annotated_clip.vhdr", "GOBLELIDSTONE~_1f68f041-84ac-41b0-be3d-638d17500c7a_annotations_clipped_eeg.mat"],
        ["HARGATE~ Caitl_043600c6-d543-481f-9787-e15964cc46ab_annotated_clip.vhdr", "HARGATE~ Caitl_043600c6-d543-481f-9787-e15964cc46ab_annotations_clipped_eeg.mat"],
        ["HOULT~ Liam_ee1c1462-4645-4cbc-8a8c-5377b75305e6_annotated_clip.vhdr", "HOULT~ Liam_ee1c1462-4645-4cbc-8a8c-5377b75305e6_annotations_clipped_eeg.mat"],
        ["MABIOR~ Diing_086c6be5-b0a5-4093-8d87-3cdd7e22a966_annotated_clip.vhdr", "MABIOR~ Diing_086c6be5-b0a5-4093-8d87-3cdd7e22a966_annotations_clipped_eeg.mat"],
        ["MACGREGOR~ Sop_6d81a4d7-a9e5-48dd-aed6-b16e3c30cb18_annotated_clip.vhdr", "MACGREGOR~ Sop_6d81a4d7-a9e5-48dd-aed6-b16e3c30cb18_annotations_clipped_eeg.mat"],
        ["MARTIN~ Scott_613958e7-a4a1-44c6-942c-7110b68bc6da_annotated_clip.vhdr", "MARTIN~ Scott_613958e7-a4a1-44c6-942c-7110b68bc6da_annotations_clipped_eeg.mat"],
        ["MASRI~ Malik_e71a0a85-c927-4d58-acc8-649399a953a3_annotated_clip.vhdr", "MASRI~ Malik_e71a0a85-c927-4d58-acc8-649399a953a3_annotations_clipped_eeg.mat"],
        ["MATHESON~ Finl_cc1379e2-9838-4a2e-b8c2-c9f99cc4e8bc_annotated_clip.vhdr", "MATHESON~ Finl_cc1379e2-9838-4a2e-b8c2-c9f99cc4e8bc_annotations_clipped_eeg.mat"],
        ["McELLIGOTT~ Ev_400b9479-5a48-4712-abd6-7b9ef449881b_annotated_clip.vhdr", "McELLIGOTT~ Ev_400b9479-5a48-4712-abd6-7b9ef449881b_annotations_clipped_eeg.mat"],
        ["OKEEFFE~ Emma_4595d807-92d7-4cb7-a36a-bfc1df8afd24_annotated_clip.vhdr", "OKEEFFE~ Emma_4595d807-92d7-4cb7-a36a-bfc1df8afd24_annotations_clipped_eeg.mat"],
        ["OLSON~ Andrew_16027753-5808-4498-aed8-966f29a04a04_annotated_clip.vhdr", "OLSON~ Andrew_16027753-5808-4498-aed8-966f29a04a04_annotations_clipped_eeg.mat"],
        ["RAMIREZ~ Amand_91bd3985-137c-405d-a532-b403f7c94705_annotated_clip.vhdr", "RAMIREZ~ Amand_91bd3985-137c-405d-a532-b403f7c94705_annotations_clipped_eeg.mat"],
        ["SALMON~ Olivia_7e208bf8-76e4-4217-a472-5e2de8b92baf_annotated_clip.vhdr", "SALMON~ Olivia_7e208bf8-76e4-4217-a472-5e2de8b92baf_annotations_clipped_eeg.mat"],
        ["SINJAR~ Majd_c7c32ec6-597b-494b-b451-ad6960720c98_annotated_clip.vhdr", "SINJAR~ Majd_c7c32ec6-597b-494b-b451-ad6960720c98_annotations_clipped_eeg.mat"],
        ["TORRIE~ Brookl_50ddff17-9579-4b28-874f-756be76b6c19_annotated_clip.vhdr", "TORRIE~ Brookl_50ddff17-9579-4b28-874f-756be76b6c19_annotations_clipped_eeg.mat"],
        ["WASYLENKA~ Dyl_0f73ce29-1914-4169-98d7-4613433ac73e_annotated_clip.vhdr", "WASYLENKA~ Dyl_0f73ce29-1914-4169-98d7-4613433ac73e_annotations_clipped_eeg.mat"],
        ["WHALEN~ Riley_8780cda3-ccb4-4918-8c25-3f6f73798886_annotated_clip.vhdr", "WHALEN~ Riley_8780cda3-ccb4-4918-8c25-3f6f73798886_annotations_clipped_eeg.mat"],
        ["WICKHORST~ Jos_a67e52c8-45ab-4a94-9550-66eef67c9c90_annotated_clip.vhdr", "WICKHORST~ Jos_a67e52c8-45ab-4a94-9550-66eef67c9c90_annotations_clipped_eeg.mat"],
        ["WILSON~ Jack_973d043c-c443-4c62-8bfc-07697e98951d_annotated_clip.vhdr", "WILSON~ Jack_973d043c-c443-4c62-8bfc-07697e98951d_annotations_clipped_eeg.mat"],
        ["WOOD~ Dawson_a6c0af5a-b1c7-47fe-8638-e4199ae768d2_annotated_clip.vhdr", "WOOD~ Dawson_a6c0af5a-b1c7-47fe-8638-e4199ae768d2_annotations_clipped_eeg.mat"],
        ]

        return dataset_name, data_path, eeg_files_info
    
    def Ree_11Pats_Validation_July2024(self):
        dataset_name = "Ree_11Pats_Validation_July2024"       
        data_path = ""
        if socket.gethostname() == "LAPTOP-TFQFNF6U":
            data_path = "D:/Ree_Files_July_2024/"
        elif socket.gethostname() == "DLP":
            data_path = "E:/Ree_Files_July_2024/"
        
        # eeg file, annotation file
        eeg_files_info = \
        [            
        ["ALI~ Ziya_Clipped.edf","_"],
        ["AUBE~ Lincoln_Clipped.edf","_"],
        ["CAMARA~ Isla_Clipped.edf","_"],
        ["ELRAFIH~ Musta_Clipped.edf","_"],
        ["HASSANIN~ Yasmine_Clipped.edf","_"],
        ["LACOMBE~ Luke_Clipped.edf","_"],
        ["LECERF~ William_Clipped.edf","_"],
        ["MOUSSA~ Noora_Clipped.edf","_"],
        ["ODLAND~ Ember_Clipped.edf","_"],
        ["ROSEVEAR~ George_Clipped.edf","_"],
        ["VANBOVEN~ Gerrit_Clipped.edf","_"],
        ]

        return dataset_name, data_path, eeg_files_info

if __name__ == "__main__":
    StudiesInfo().Maggi_Stroke_Data_1()
    StudiesInfo().Maggi_Stroke_Data_2()
    StudiesInfo().Maggi_Stroke_AES2024_2ndPart()
    StudiesInfo().Minette_AED()
    StudiesInfo().Multidetect_Validated_HFO()
    StudiesInfo().Ree_11Pats_Validation_July2024()

    pass
