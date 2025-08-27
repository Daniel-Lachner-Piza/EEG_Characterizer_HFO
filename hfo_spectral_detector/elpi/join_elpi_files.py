import mne
import numpy as np
import pandas as pd
import os
import datetime

from hfo_spectral_detector.studies_info.studies_info import StudiesInfo
from hfo_spectral_detector.elpi.elpi_interface import load_elpi_file, write_elpi_file, get_agreement_between_elpi_files

def get_shortest_and_singles(elpi_annots_a, elpi_annots_b):
    
    mtg_labels = np.unique(np.concatenate([elpi_annots_a.Channel.unique(), elpi_annots_b.Channel.unique()]))

    elpi_annots_a = elpi_annots_a.sort_values(by=['Channel', 'StartSec']).reset_index(drop=True)
    elpi_annots_b.sort_values(by=['Channel', 'StartSec']).reset_index(drop=True)

    ch_pruned_events = pd.DataFrame()

    for idx_a, event_a in elpi_annots_a.iterrows():
        chann_a = event_a.Channel
        center_a = (event_a.StartSample + event_a.EndSample)/2
        duration_a = event_a.EndSample - event_a.StartSample
        touches = 0
        for idx_b, event_b in elpi_annots_b.iterrows():
            chann_b = event_b.Channel
            if chann_a == chann_b:
                duration_b = event_b.EndSample - event_b.StartSample
                overlap_a = (event_a.StartSample>=event_b.StartSample) & (event_a.StartSample<=event_b.EndSample)
                overlap_b = (event_a.EndSample>=event_b.StartSample) & (event_a.EndSample<=event_b.EndSample)
                overlap_c = (event_a.StartSample<=event_b.StartSample) & (event_a.EndSample>=event_b.EndSample)
                overlap_d = (center_a >=event_b.StartSample) & (center_a <=event_b.EndSample)
                annotation_overlap_ok = overlap_a or overlap_b or overlap_c or overlap_d
                if annotation_overlap_ok:
                    if duration_a <= duration_b:
                        ch_pruned_events = pd.concat([ch_pruned_events, event_a.to_frame().T], ignore_index=True)
                    else:
                        ch_pruned_events = pd.concat([ch_pruned_events, event_b.to_frame().T], ignore_index=True)
                    touches += 1
                    if touches > 1:
                        pass
                pass
        if touches == 0:
            ch_pruned_events = pd.concat([ch_pruned_events, event_a.to_frame().T], ignore_index=True)

    return ch_pruned_events

def get_singles(elpi_annots_a, elpi_annots_b):
    
    mtg_labels = np.unique(np.concatenate([elpi_annots_a.Channel.unique(), elpi_annots_b.Channel.unique()]))

    elpi_annots_a = elpi_annots_a.sort_values(by=['Channel', 'StartSec']).reset_index(drop=True)
    elpi_annots_b.sort_values(by=['Channel', 'StartSec']).reset_index(drop=True)

    ch_pruned_events = pd.DataFrame()

    for idx_a, event_a in elpi_annots_a.iterrows():
        chann_a = event_a.Channel
        center_a = (event_a.StartSample + event_a.EndSample)/2
        duration_a = event_a.EndSample - event_a.StartSample
        touches = 0
        for idx_b, event_b in elpi_annots_b.iterrows():
            chann_b = event_b.Channel
            if chann_a == chann_b:
                duration_b = event_b.EndSample - event_b.StartSample
                overlap_a = (event_a.StartSample>=event_b.StartSample) & (event_a.StartSample<=event_b.EndSample)
                overlap_b = (event_a.EndSample>=event_b.StartSample) & (event_a.EndSample<=event_b.EndSample)
                overlap_c = (event_a.StartSample<=event_b.StartSample) & (event_a.EndSample>=event_b.EndSample)
                overlap_d = (center_a >=event_b.StartSample) & (center_a <=event_b.EndSample)
                annotation_overlap_ok = overlap_a or overlap_b or overlap_c or overlap_d
                if annotation_overlap_ok:
                    touches += 1
                    if touches > 1:
                        pass
                pass
        if touches == 0:
            ch_pruned_events = pd.concat([ch_pruned_events, event_a.to_frame().T], ignore_index=True)

    return ch_pruned_events

if __name__ == "__main__":

    dataset_name, data_path, eeg_files_info = StudiesInfo().Maggi_11Pats_Validation_Dez2024_PostHoc()
    eeg_fn_ls = [fi[0] for fi in eeg_files_info]
    visual_marks_ls = [fi[1] for fi in eeg_files_info]
    for f1, infos in enumerate(zip(eeg_fn_ls, visual_marks_ls)):
        eeg_fpath = infos[0]
        visual_marks_fpath = infos[1]
        
        print(eeg_fpath)

        vm_fpath_a = data_path + visual_marks_fpath
        vm_fpath_a = vm_fpath_a.replace("_merged_pruned", "_best_performer_detections")

        vm_fpath_b = vm_fpath_a.replace("_best_performer_detections", "_maggi_annotations_corrected")
        
        elpi_annots_a = load_elpi_file(vm_fpath_a)
        elpi_annots_b = load_elpi_file(vm_fpath_b)

        assert(len(elpi_annots_a) > 0), f"Empty file: {vm_fpath_a}"
        assert(len(elpi_annots_b) > 0), f"Empty file: {vm_fpath_b}"

        shortest_and_singles_df = get_shortest_and_singles(elpi_annots_a.copy(), elpi_annots_b.copy())
        shortest_and_singles_df = get_shortest_and_singles(shortest_and_singles_df.copy(), shortest_and_singles_df.copy())

        singles_df = get_singles(elpi_annots_b.copy(), shortest_and_singles_df.copy())
        
        pruned_events_df = pd.concat([shortest_and_singles_df, singles_df]).sort_values(by=['Channel', 'StartSec']).reset_index(drop=True)
        pruned_events_df.Type = pruned_events_df.Type.strip()

        pruned_events_fpath = vm_fpath_a.replace("_best_performer_detections", "_merged_pruned")
        write_elpi_file(pruned_events_df, pruned_events_fpath)