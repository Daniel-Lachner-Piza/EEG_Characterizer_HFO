import scipy.io as sio
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, f1_score, recall_score, recall_score, precision_score, accuracy_score, fbeta_score



def load_elpi_file(mat_fname:str=None) -> pd.DataFrame:

    if os.path.isfile(mat_fname):
        mat_contents = sio.loadmat(mat_fname)
        mat_file_columns = mat_contents.keys()
        if 'detections' in mat_file_columns:
            detections_data = mat_contents['detections']
            nr_annots = detections_data.shape[0]

            elpi_annots_df = pd.DataFrame(data={'Channel':[detection[0][0] for detection in detections_data]}) 
            elpi_annots_df['Type'] = [detection[1][0] for detection in detections_data]
            elpi_annots_df['StartSec'] = [detection[2][0][0] for detection in detections_data]
            elpi_annots_df['EndSec'] = [detection[3][0][0] for detection in detections_data]
            elpi_annots_df['StartSample'] = [detection[4][0][0] for detection in detections_data]
            elpi_annots_df['EndSample'] = [detection[5][0][0] for detection in detections_data]
            elpi_annots_df['Comments'] = ["_"]*nr_annots
            elpi_annots_df['ChSpec'] = [detection[7][0][0] for detection in detections_data]
            elpi_annots_df['CreationTime'] = [detection[8][0] for detection in detections_data]
            elpi_annots_df['User'] = ["_"]*nr_annots
        # elif ('Type' in mat_file_columns) and ('StartSample' in mat_file_columns) and ('ChSpec' in mat_file_columns):
        #     elpi_annots_df = pd.DataFrame(data={'Channel':[str(val[0][0]) for val in mat_contents['Channel']]})
        #     elpi_annots_df['Type'] = [str(val[0][0]) for val in mat_contents['Type']]
        #     elpi_annots_df['StartSec'] = [val[0][0][0] for val in mat_contents['StartSec']]
        #     elpi_annots_df['EndSec'] = [val[0][0][0] for val in mat_contents['EndSec']]
        #     elpi_annots_df['StartSample'] = [val[0][0][0] for val in mat_contents['StartSample']]
        #     elpi_annots_df['EndSample'] = [val[0][0][0] for val in mat_contents['EndSample']]
        #     elpi_annots_df['Comments'] = ['']*len([str(val[0][0]) for val in mat_contents['Type']])
        #     elpi_annots_df['ChSpec'] = [val[0][0][0] for val in mat_contents['ChSpec']]
        #     elpi_annots_df['CreationTime'] = [str(val[0][0]) for val in mat_contents['CreationTime']]
        #     elpi_annots_df['User'] = [str(val) for val in mat_contents['User']]
        elif ('Type' in mat_file_columns) and ('StartSample' in mat_file_columns) and ('ChSpec' in mat_file_columns):
            elpi_annots_df = pd.DataFrame(data={'Channel':mat_contents['Channel'].flatten()})
            elpi_annots_df['Type'] = mat_contents['Type'].flatten()
            elpi_annots_df['StartSec'] = mat_contents['StartSec'].flatten()
            elpi_annots_df['EndSec'] = mat_contents['EndSec'].flatten()
            elpi_annots_df['StartSample'] = mat_contents['StartSample'].flatten()
            elpi_annots_df['EndSample'] = mat_contents['EndSample'].flatten()
            elpi_annots_df['Comments'] = mat_contents['Comments'].flatten()
            elpi_annots_df['ChSpec'] = mat_contents['ChSpec'].flatten()
            elpi_annots_df['CreationTime'] = mat_contents['CreationTime'].flatten()
            elpi_annots_df['User'] = mat_contents['User'].flatten()

    else:
        elpi_annots_df = pd.DataFrame(data={'Channel':["_"]})
        elpi_annots_df['Type'] = ["_"]
        elpi_annots_df['StartSec'] = [0]
        elpi_annots_df['EndSec'] = [0]
        elpi_annots_df['StartSample'] = [0]
        elpi_annots_df['EndSample'] = [0]
        elpi_annots_df['Comments'] = ["_"]
        elpi_annots_df['ChSpec'] = [0]
        elpi_annots_df['CreationTime'] = ["_"]
        elpi_annots_df['User'] = ["_"]

    # Watch out, sio.savemat adds blank spaces to the strings!
    # Remove these blank spaces here
    elpi_annots_df['Type'] = elpi_annots_df['Type'].apply(str.strip)
    elpi_annots_df['Channel'] = elpi_annots_df['Channel'].apply(str.strip)
    return elpi_annots_df

def write_elpi_file(annots_df, mat_fname):

    # Watch out, sio.savemat adds blank spaces to the strings!
    sio.savemat(mat_fname, annots_df.to_dict(orient='list'), oned_as='column')
    pass

def get_agreement_event_wise(gs_marks, predicted_marks, mtg_labels, eeg_dur_s):
        

    sensitivity = 0
    precision = 0
    f1_val = 0
    gs_cnts = 0
    detects_cnt = 0
    performance_chwise = {'Channel':[], 'Sensitivity':[], 'Specificity':[],'Precision':[], 'Kappa':[], 'MCC':[], 'F1':[]}
    for mtg in mtg_labels:
        print(mtg)
        if mtg.lower() == "fz-cz" or mtg.lower() == "cz-pz":
            continue
        ch_gs_marks = gs_marks[gs_marks.Channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)        
        ch_pred_marks = predicted_marks[predicted_marks.Channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)

        if len(ch_gs_marks) == 0:
            continue

        gs_cnts += len(ch_gs_marks)
        detects_cnt += len(ch_pred_marks)

        gs_ch_touched = np.zeros(len(ch_gs_marks), dtype=bool)
        pred_ch_touched = np.zeros(len(ch_pred_marks), dtype=bool)

        for gs_idx, gs_mark in ch_gs_marks.iterrows():
            gs_center = gs_mark.StartSec + (gs_mark.EndSec-gs_mark.StartSec)/2
            for pred_idx, pred_mark in ch_pred_marks.iterrows():
                if gs_center > pred_mark.StartSec and gs_center < pred_mark.EndSec:
                    gs_ch_touched[gs_idx] = True
                    pred_ch_touched[pred_idx] = True

        for pred_idx, pred_mark in ch_pred_marks.iterrows():
            pred_center = pred_mark.StartSec + (pred_mark.EndSec-pred_mark.StartSec)/2
            for gs_idx, gs_mark in ch_gs_marks.iterrows():
                if pred_center > gs_mark.StartSec and pred_center < gs_mark.EndSec:
                    pred_ch_touched[pred_idx] = True
                    gs_ch_touched[gs_idx] = True

        nr_event_wdws = eeg_dur_s/0.05
        nr_negs = nr_event_wdws - len(ch_gs_marks)
        tp = np.sum(pred_ch_touched)
        fp = np.sum(np.logical_not(pred_ch_touched))
        fn = np.sum(np.logical_not(gs_ch_touched))
        tn = nr_event_wdws - len(ch_pred_marks) - fn
        specificity_ch = tn/(tn+fp)
        matthews_corrcoef_ch = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        kapp_a = 2 * (tp*tn-fn*fp)
        kappa_b = (tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)
        kappa_val_ch =  kapp_a/kappa_b

        sensitivity_ch = np.sum(gs_ch_touched)/len(gs_ch_touched)
        precision_ch = np.sum(pred_ch_touched)/len(pred_ch_touched)
        f1_ch = 2*precision_ch*sensitivity_ch/(precision_ch+sensitivity_ch)
        performance_chwise['Channel'].append(mtg)
        performance_chwise['Sensitivity'].append(sensitivity_ch)
        performance_chwise['Specificity'].append(specificity_ch)
        performance_chwise['Precision'].append(precision_ch)
        performance_chwise['Kappa'].append(kappa_val_ch)
        performance_chwise['MCC'].append(matthews_corrcoef_ch)
        performance_chwise['F1'].append(f1_ch)

        #print("SensCh: ", sensitivity_ch, "SpecCh: ", specificity_ch, "PrecCh: ", precision_ch, "KappaCh: ", kappa_val_ch)

    performance_chwise_df = pd.DataFrame(performance_chwise)
    sensitivity = performance_chwise_df.Sensitivity.mean()
    specificity = performance_chwise_df.Specificity.mean()
    precision = performance_chwise_df.Precision.mean()
    kappa_val = performance_chwise_df.Kappa.mean()
    mcc_val = performance_chwise_df.MCC.mean()
    f1_val = performance_chwise_df.F1.mean()

    # assert gs_cnts == len(gs_marks), "Something went wrong during the assignment of visually-validated-events to the EEG channels"
    # assert detects_cnt == len(predicted_marks), "Something went wrong during the assignment of detections to the EEG channels"

    return sensitivity, specificity, precision, kappa_val, mcc_val, f1_val

def get_metrics_dlp(gs_marks, predicted_marks, mtg_labels, eeg_dur_s):

    valid_eeg_channels = [ch.lower() for ch in gs_marks.Channel.unique()]
    gold_standard = gs_marks
    prediction = predicted_marks
    gs_touched_cntr = 0
    for ri in np.arange(len(gold_standard)):
        this_annot_start_sec = gold_standard.at[ri,'StartSec']
        this_annot_end_sec = gold_standard.at[ri,'EndSec']
        this_annot_center_sec = np.mean([this_annot_start_sec, this_annot_end_sec])
        this_annot_ch = gold_standard.at[ri,'Channel']
        if this_annot_ch.lower() not in valid_eeg_channels:
            continue

        detects_sel_ch = prediction.Channel.str.fullmatch(this_annot_ch, case=False)                
        ch_detects_df = prediction[detects_sel_ch]
        overlap_a = (this_annot_start_sec>=ch_detects_df['StartSec']) & (this_annot_start_sec<=ch_detects_df['EndSec'])
        overlap_b = (this_annot_end_sec>=ch_detects_df['StartSec']) & (this_annot_end_sec<=ch_detects_df['EndSec'])
        overlap_c = (this_annot_start_sec<=ch_detects_df['StartSec']) & (this_annot_end_sec>=ch_detects_df['EndSec'])
        overlap_d = (this_annot_center_sec>=ch_detects_df['StartSec']) & (this_annot_center_sec<=ch_detects_df['EndSec'])
        annotation_overlap_ok = sum(overlap_a.to_numpy())>0 or sum(overlap_b.to_numpy())>0 or sum(overlap_c.to_numpy())>0 or sum(overlap_d.to_numpy())>0
        gs_touched_cntr += annotation_overlap_ok
    
    gold_standard = predicted_marks
    prediction = gs_marks
    pred_touched_cntr = 0
    for ri in np.arange(len(gold_standard)):
        this_annot_start_sec = gold_standard.at[ri,'StartSec']
        this_annot_end_sec = gold_standard.at[ri,'EndSec']
        this_annot_center_sec = np.mean([this_annot_start_sec, this_annot_end_sec])
        this_annot_ch = gold_standard.at[ri,'Channel']
        if this_annot_ch.lower() not in valid_eeg_channels:
            continue

        detects_sel_ch = prediction.Channel.str.fullmatch(this_annot_ch, case=False)                
        ch_detects_df = prediction[detects_sel_ch]
        overlap_a = (this_annot_start_sec>=ch_detects_df['StartSec']) & (this_annot_start_sec<=ch_detects_df['EndSec'])
        overlap_b = (this_annot_end_sec>=ch_detects_df['StartSec']) & (this_annot_end_sec<=ch_detects_df['EndSec'])
        overlap_c = (this_annot_start_sec<=ch_detects_df['StartSec']) & (this_annot_end_sec>=ch_detects_df['EndSec'])
        overlap_d = (this_annot_center_sec>=ch_detects_df['StartSec']) & (this_annot_center_sec<=ch_detects_df['EndSec'])
        annotation_overlap_ok = sum(overlap_a.to_numpy())>0 or sum(overlap_b.to_numpy())>0 or sum(overlap_c.to_numpy())>0 or sum(overlap_d.to_numpy())>0
        pred_touched_cntr += annotation_overlap_ok

    nr_event_wdws = eeg_dur_s/0.05
    nr_channs = len(valid_eeg_channels)
    nr_negs = (nr_event_wdws - len(gs_marks))*nr_channs
    tp = pred_touched_cntr
    fp = len(predicted_marks) - pred_touched_cntr
    fn = len(gs_marks) - gs_touched_cntr
    tn = nr_event_wdws - (len(predicted_marks) - fn)

    specificity = tn/(tn+fp)
    mcc_val = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    kapp_a = 2 * (tp*tn-fn*fp)
    kappa_b = (tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)
    kappa_val =  kapp_a/kappa_b
    sensitivity = gs_touched_cntr/len(gs_marks)
    precision = pred_touched_cntr/len(predicted_marks)
    f1_val = 2*precision*sensitivity/(precision+sensitivity)

    return sensitivity, specificity, precision, kappa_val, mcc_val, f1_val




def get_agreement_between_elpi_files(gs_marks, predicted_marks, mtg_labels, eeg_dur_s, mask_step_size_ms:int=1):
    
    nr_wdws = int(np.round((eeg_dur_s*1000)/mask_step_size_ms))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    sensitivity = 0
    specificity = 0
    precision = 0
    kappa_val = 0
    mcc_val = 0
    f1_val = 0

    for idx, mtg in enumerate(mtg_labels):
            
        
        ch_gs_marks = gs_marks[gs_marks.Channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)        
        ch_pred_marks = predicted_marks[predicted_marks.Channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)

        if len(ch_pred_marks) > 0 and len(ch_gs_marks) == 0:
            pass

        eoi_gs_mask = np.zeros(nr_wdws, dtype=bool)
        if len(ch_gs_marks)>0:
            nr_eoi = len(ch_gs_marks)
            eoi_start_ms = np.round((ch_gs_marks.StartSec.to_numpy()*1000)/mask_step_size_ms).astype(np.int64)
            eoi_end_ms = np.round((ch_gs_marks.EndSec.to_numpy()*1000)/mask_step_size_ms).astype(np.int64)
            for idx in np.arange(nr_eoi):
                eoi_gs_mask[eoi_start_ms[idx]:eoi_end_ms[idx]] = True
            
        eoi_pred_mask = np.zeros(nr_wdws, dtype=bool)
        if len(ch_pred_marks)>0:
            nr_eoi = len(ch_pred_marks)
            eoi_start_ms = np.round((ch_pred_marks.StartSec.to_numpy()*1000)/mask_step_size_ms).astype(np.int64)
            eoi_end_ms = np.round((ch_pred_marks.EndSec.to_numpy()*1000)/mask_step_size_ms).astype(np.int64)
            for idx in np.arange(nr_eoi):
                eoi_pred_mask[eoi_start_ms[idx]:eoi_end_ms[idx]] = True
            
        # Self calculation of performance metrics
        # tp += np.sum(np.logical_and(eoi_gs_mask, eoi_pred_mask))
        # tn += np.sum(np.logical_and(np.logical_not(eoi_gs_mask), np.logical_not(eoi_pred_mask)))
        # fp += np.sum(np.logical_and(np.logical_not(eoi_gs_mask), eoi_pred_mask))
        # fn += np.sum(np.logical_and(eoi_gs_mask, np.logical_not(eoi_pred_mask)))

        # sensitivity_ch = tp/(tp+fn) 
        # specificity_ch = tn/(tn+fp) 
        # precision_ch = tp/(tp+fp)
        # kapp_a = 2 * (tp*tn-fn*fp)
        # kappa_b = (tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)
        # kappa_val_ch =  kapp_a/kappa_b

        y_gs = eoi_gs_mask
        y_pred = eoi_pred_mask
        specificity_val_ch_sk = recall_score(y_gs, y_pred, pos_label=0)
        recall_val_ch_sk = recall_score(y_gs, y_pred)
        precision_val_ch_sk = precision_score(y_gs, y_pred)
        accuracy_val_ch_sk = accuracy_score(y_gs, y_pred)
        f1_val_ch_sk = f1_score(y_gs, y_pred)
        mcc_val_ch_sk = matthews_corrcoef(y_gs, y_pred)
        kappa_val_ch_sk = cohen_kappa_score(y_gs, y_pred)

        sensitivity += recall_val_ch_sk
        specificity += specificity_val_ch_sk
        precision += precision_val_ch_sk
        kappa_val += kappa_val_ch_sk
        mcc_val += mcc_val_ch_sk
        f1_val += f1_val_ch_sk
        pass

    sensitivity /= len(mtg_labels)
    specificity /= len(mtg_labels)
    precision /= len(mtg_labels)
    kappa_val /= len(mtg_labels)
    mcc_val /= len(mtg_labels)
    f1_val /= len(mtg_labels)

    return sensitivity, specificity, precision, kappa_val, mcc_val, f1_val


def unifiy_events(annots_fn, eeg_dur_s, fs):
    
    elpi_annots_df = load_elpi_file(annots_fn)
    
    # mask_step_size_ms = 1
    # time_mask = np.arange(0, eeg_dur_s*1000, mask_step_size_ms)
    # mtg_labels = annots_df.Channel.unique()
    # for idx, mtg in enumerate(mtg_labels):

    #     this_ch_objs_df = elpi_annots[elpi_annots.Channel.str.lower()==mtg.lower()]
    #     this_ch_objs_df = elpi_annots[elpi_annots.Channel.str.fullmatch(mtg.lower(), case=False)].reset_index(drop=True)
             
    #     if len(this_ch_objs_df)>0:
    #         chan_eoi_start = this_ch_objs_df.start_ms.to_numpy()
    #         chan_eoi_end = this_ch_objs_df.end_ms.to_numpy()
    #         chan_eoi_period_s = 1000/this_ch_objs_df.freq_centroid_Hz.to_numpy()
    #         chan_eoi_start_ms -= chan_eoi_period_s
    #         chan_eoi_end_ms += chan_eoi_period_s

    #         # Create a zeroed-mask for all samples
    #         # If an EOI is found for a certain sample range, then place a 1 within this range
    #         eoi_mask = np.zeros_like(time_mask)
    #         for idx, (eoi_start_ms, eoi_end_ms) in enumerate(zip(chan_eoi_start_ms, chan_eoi_end_ms)):
    #             eoi_mask[np.logical_and(time_mask>=eoi_start_ms,time_mask<=eoi_end_ms)] = 1

    #         # Get the start and end times of the zeroed-mask by finding the first and last 1 from sequences surrounded by zeros
    #         eoi_start_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == 1)+1]
    #         eoi_end_times_ms = time_mask[np.argwhere(np.diff(eoi_mask) == -1)]
    #         eoi_start_times_ms = eoi_start_times_ms.flatten()
    #         eoi_end_times_ms = eoi_end_times_ms.flatten()
    #         eoi_durations_ms = eoi_end_times_ms-eoi_start_times_ms
    #         eoi_start_samples = np.round(eeg_fs*eoi_start_times_ms/1000)
    #         eoi_start_samples = eoi_start_samples.astype(np.int64)
    #         eoi_end_samples = np.round(eeg_fs*eoi_end_times_ms/1000)
    #         eoi_end_samples = eoi_end_samples.astype(np.int64)

    #         nr_eoi = len(eoi_start_times_ms)

    #         # Create a dictionary with the EOI information so that it can be read in Elpi
    #         creation_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    #         merged_eoi_dict = {
    #             "Channel": [mtg] * nr_eoi,
    #             "Type": ["spect_HFO"] * nr_eoi,
    #             "StartSec": eoi_start_times_ms/1000,
    #             "EndSec": eoi_end_times_ms/1000,
    #             "StartSample": eoi_start_samples,
    #             "EndSample": eoi_end_samples,
    #             "Comments": [pat_name] * nr_eoi,
    #             "ChSpec": np.ones(nr_eoi, dtype=bool),
    #             "CreationTime": [creation_time] * nr_eoi,
    #             "User": ["DLP_Prune_HFO"] * nr_eoi,
    #         }

    #         # Check that all columns have the same number of elements
    #         #print([len(value) for key, value in merged_eoi_dict.items()])

    #         if nr_eoi > 0:
    #             for key, values in merged_eoi_dict.items():
    #                 assert (key in all_channs_eoi_dict.keys()), "Key not found in all channels dictionary!"
    #                 if key in all_channs_eoi_dict.keys():
    #                     all_channs_eoi_dict[key].extend(values)
    #     else:
    #         print("No EOI in channel: ", mtg)

    #     #detections_mat_fn = (out_files_dest_path + f"{pat_name}_spectrogram_HFO.mat")
    #     #savemat(detections_mat_fn, all_channs_eoi_dict)
    #     pass
