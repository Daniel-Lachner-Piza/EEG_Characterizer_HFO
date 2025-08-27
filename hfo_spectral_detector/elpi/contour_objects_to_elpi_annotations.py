import mne
import numpy as np
import pandas as pd
import os
import datetime
from scipy.io import savemat

def save_contour_objects_as_elpi_annots(eeg_filepath, all_ch_contours_df, elpi_files_dest_path):

    eeg_filename = eeg_filepath.split('/')[-1]
    pat_name = eeg_filename.replace(".edf",'')

    eeg_data = mne.io.read_raw_edf(eeg_filepath, verbose=False)
    fs = eeg_data.info["sfreq"]
    n_samples = eeg_data.n_times
    eeg_dur_ms = 1000*n_samples/fs
    eeg_nr_mins = n_samples/fs/60
    assert fs > 1000, "Sampling Rate is under 1000 Hz!"


    subj_occ_rates = {"Subject": [], "Channel": [], "NI": [], "OccRate": []}        
    all_channs_eoi_dict = {
        "Channel": [],
        "Type": [],
        "StartSec": [],
        "EndSec": [],
        "StartSample": [],
        "EndSample": [],
        "Comments": [],
        "ChSpec": [],
        "CreationTime": [],
        "User": [],
    }

    all_mtgs = np.unique(all_ch_contours_df.channel)
    for idx, mtg in enumerate(all_mtgs):
        this_ch_objs_df = all_ch_contours_df[all_ch_contours_df.channel==mtg]

        this_ch_objs_df = this_ch_objs_df[np.logical_or(this_ch_objs_df.visual_valid, np.logical_and(this_ch_objs_df.spect_ok, this_ch_objs_df.bp_ok))]

        chan_eoi_start_ms = this_ch_objs_df.start_ms.to_numpy()
        chan_eoi_end_ms = this_ch_objs_df.end_ms.to_numpy()
        chan_eoi_period_s = 1000/this_ch_objs_df.freq_centroid_Hz.to_numpy()
        chan_eoi_start_ms -= chan_eoi_period_s
        chan_eoi_end_ms += chan_eoi_period_s

        nr_eoi = 0

        # Create a zeroed-mask for all samples
        # If an EOI is found for a certain sample range, then place a 1 within this range
        mask_step_size_ms = 1
        time_mask = np.arange(0, eeg_dur_ms, mask_step_size_ms)
        eoi_mask = np.zeros_like(time_mask)
        for idx, (eoi_start_ms, eoi_end_ms) in enumerate(zip(chan_eoi_start_ms, chan_eoi_end_ms)):
            eoi_mask[np.logical_and(time_mask>=eoi_start_ms,time_mask<=eoi_end_ms)] = 1

        # Get the start and end times of the zeroed-mask by finding the first and last 1 from sequences surrounded by zeros
        eoi_start_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == 1)+1]
        eoi_end_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == -1)]
        eoi_start_times_ms = eoi_start_times_ms.flatten()
        eoi_end_times_ms = eoi_end_times_ms.flatten()
        eoi_durations_ms = eoi_end_times_ms-eoi_start_times_ms
        eoi_start_samples = np.round(fs*eoi_start_times_ms/1000)
        eoi_start_samples = eoi_start_samples.astype(np.int64)
        eoi_end_samples = np.round(fs*eoi_end_times_ms/1000)
        eoi_end_samples = eoi_end_samples.astype(np.int64)

        nr_eoi = len(eoi_start_times_ms)
        # Create a dictionary with the EOI information so that it can be read in Elpi
        creation_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        merged_eoi_dict = {
            "Channel": [mtg] * nr_eoi,
            "Type": ["spect_HFO"] * nr_eoi,
            "StartSec": eoi_start_times_ms/1000,
            "EndSec": eoi_end_times_ms/1000,
            "StartSample": eoi_start_samples,
            "EndSample": eoi_end_samples,
            "Comments": [pat_name] * nr_eoi,
            "ChSpec": np.ones(nr_eoi, dtype=bool),
            "CreationTime": [creation_time] * nr_eoi,
            "User": ["DLP_Prune_HFO"] * nr_eoi,
        }

        print([len(value) for key, value in merged_eoi_dict.items()])

        if nr_eoi > 0:
            for key, values in merged_eoi_dict.items():
                assert (key in all_channs_eoi_dict.keys()), "Key not found in all channels dictionary!"
                if key in all_channs_eoi_dict.keys():
                    all_channs_eoi_dict[key].extend(values)

        # Get occ.rate
        ch_occ_rate = nr_eoi / eeg_nr_mins
        subj_occ_rates["Subject"].append(pat_name)
        subj_occ_rates["Channel"].append(mtg)
        subj_occ_rates["OccRate"].append(ch_occ_rate)
        subj_occ_rates["NI"].append(0)

    detections_mat_fn = (elpi_files_dest_path + f"{eeg_filename}_spectrogram_HFO.mat")
    savemat(detections_mat_fn, all_channs_eoi_dict)


if __name__ == "__main__":

    data_path = "F:/Postdoc_Calgary/Research/Physiological_HFO/Ree_Files_July_2024/"

    # eeg file, annotation file
    eeg_files_info = \
    [
    ["LECERF~ William_Clipped.edf","_"],
    ["MOUSSA~ Noora_Clipped.edf","_"],
    ["ODLAND~ Ember_Clipped.edf","_"],
    ["ROSEVEAR~ George_Clipped.edf","_"],
    ["VANBOVEN~ Gerrit_Clipped.edf","_"],
    ["ALI~ Ziya_Clipped.edf","_"],
    ["AUBE~ Lincoln_Clipped.edf","_"],
    ["CAMARA~ Isla_Clipped.edf","_"],
    ["ELRAFIH~ Musta_Clipped.edf","_"],
    ["HASSANIN~ Yasmine_Clipped.edf","_"],
    ["LACOMBE~ Luke_Clipped.edf","_"],
    ]

    # Define Output Path
    out_path = "H:/Scalp_HFO_Spect_Detector" + os.sep + "Ree_Output"+ os.sep
    elpi_files_dest_path = out_path + "elpi_detections"+ os.sep
    os.makedirs(elpi_files_dest_path, exist_ok=True)

    for eeg_fn, elpi_annotations_fn in eeg_files_info:

        pat_name = eeg_fn.replace(".edf",'')

        # Read Elpi annotations
        #elpi_annots = load_gs_file(data_path+elpi_data_filename)
        # Read contour objects
        all_ch_df_filepath = out_path + pat_name + "All_Ch_Objects.csv"

        if os.path.isfile(all_ch_df_filepath):
            eeg_filepath = data_path+eeg_fn
            all_ch_contours_df = pd.read_csv(all_ch_df_filepath)
            save_contour_objects_as_elpi_annots(eeg_filepath, all_ch_contours_df, elpi_files_dest_path)
            pass

 

    pass