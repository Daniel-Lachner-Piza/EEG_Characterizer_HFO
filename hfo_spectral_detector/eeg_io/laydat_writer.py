import numpy as np
import os
import pandas as pd
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from pyeeg_toolbox.utils.io_tools import get_files_in_folder
from studies_info import ACH_Pediatric_Patients


def write_dat(eeg_data_path:str=None, pat_id:str=None, output_path:str=None):

    eeg_files_ls = get_files_in_folder(eeg_data_path, '.lay')

    this_patient_errors_df = pd.DataFrame()
    for this_pat_eeg_fpath in eeg_files_ls:
        pat_firstname= f"Patient"
        pat_lastname= f"{pat_id:05d}"
        pat_file_id_str = f"Pat{pat_id:05d}_{this_pat_eeg_fpath.name.split('-')[-1].split('.')[0]}"
        #print(pat_file_id_str)

        eeg_hdr = mne.io.read_raw_persyst(this_pat_eeg_fpath, verbose=False)

        orig_startDateTime = eeg_hdr.info['meas_date']
        orig_startDateTime = orig_startDateTime.replace(year=2024, month=9, day=15)

        calibration = float(eeg_hdr.info["chs"][0]["cal"]) # number for scaling waveform data
        calibration_persyst = calibration * 1.0e6
        elec_name_list=eeg_hdr.ch_names
        nr_channs = eeg_hdr._raw_extras[0]['n_chs']
        nr_samples = eeg_hdr._raw_extras[0]['n_samples']
        data_type = eeg_hdr._raw_extras[0]['dtype']
        first_sample_secs = eeg_hdr._raw_extras[0]['first_sample_secs']
        data_type_str = '7' # 7 is the code for int32
        if data_type != np.int32:
            data_type_str = '0' # 0 is the code for int16
        fs = eeg_hdr.info["sfreq"]

        for start_sample in np.arange(0, nr_samples, 3600*fs, dtype=int):
            end_sample = int(start_sample + 3600*fs)
            if end_sample > nr_samples:
                end_sample = nr_samples
                pass
            hr_nr = int(start_sample/3600/fs)

            startDateTime = orig_startDateTime + pd.Timedelta(seconds=start_sample/fs)
            first_sample_secs += start_sample/fs

            # Clipped EEG .dat filename
            new_dat_file_path = output_path / f"{pat_file_id_str}_h{hr_nr:03d}.dat"
            if os.path.isfile(new_dat_file_path):
                os.remove(new_dat_file_path) 
            # Clipped EEG .lay filename
            new_lay_file_path = Path(str(new_dat_file_path).replace('.dat', '.lay'))
            if os.path.isfile(new_lay_file_path):
                os.remove(new_lay_file_path)

            print(f".dat filepath: {new_dat_file_path}")
            print(f".lay filepath: {new_lay_file_path}")
            
            if not new_lay_file_path.exists():

                # Create .dat file
                #eeg_hdr = eeg_hdr.resample(sfreq=256)
                eeg_data = eeg_hdr.get_data(start=start_sample, stop=end_sample)

                # Under-sample the data
                us_fs = fs
                # us_fs = int(fs/4)
                # assert us_fs== 512, f"sampling_rate={us_fs} is not 256 Hz"
                # us_sel = np.linspace(start=0, stop=eeg_data.shape[1], num=np.round(eeg_data.shape[1]/fs*us_fs).astype(int), dtype=int, endpoint=False)
                # #signal[np.linspace(start=0, stop=eeg_data.shape[1], num=fs, dtype=int, endpoint=False)]
                # assert eeg_data.shape[1]/len(us_sel) == 4, f"eeg_data.shape[1]/len(us_sel)={eeg_data.shape[1]/len(us_sel)} is not 4"
                # eeg_data = eeg_data[:, us_sel]

                data_record = eeg_data.copy().T / calibration
                data_record = data_record.astype(data_type)
                data_record.flatten()
                with open(new_dat_file_path, 'wb') as fid:
                    data_record.tofile(fid)
                

                # Create .lay file
                new_lay_file_path = Path(str(new_dat_file_path).replace('.dat', '.lay'))
                if os.path.isfile(new_lay_file_path):
                    os.remove(new_lay_file_path)

                with open(new_lay_file_path, 'w') as lay:
                    lay.write("[FileInfo]\n")
                    lay.write(f"File={Path(new_dat_file_path).name}\n")
                    lay.write("FileType=Interleaved\n")
                    lay.write(f"SamplingRate={us_fs}\n")
                    lay.write("HeaderLength=0\n")
                    lay.write(f"Calibration={calibration_persyst}\n")
                    lay.write(f"WaveformCount={nr_channs}\n")
                    lay.write(f"DataType={data_type_str}\n")
                    lay.write(f"MainsFrequency={eeg_hdr._raw_extras[0]['mains_freq']}\n")
                    lay.write("\n")

                    lay.write("[Patient]\n");
                    lay.write(f"First={pat_firstname}\n")
                    lay.write("MI=\n")
                    lay.write(f"Last={pat_lastname}\n")
                    lay.write("Sex=\n")
                    lay.write("Hand=\n")
                    lay.write("BirthDate=1821/09/15\n")
                    lay.write(f"ID={pat_firstname}{pat_lastname}\n")
                    lay.write("MedicalRecordN=\n")
                    lay.write(f"TestDate={startDateTime.strftime("%Y.%m.%d")}\n") 
                    lay.write(f"TestTime={startDateTime.strftime("%H:%M:%S")}\n") # Note, if there are fractional seconds, you need to accommodate them in [SampleTimes]
                    lay.write("Physician=\n")
                    lay.write("Technician=\n")
                    lay.write("Medications=\n")
                    lay.write("History=\n")
                    lay.write("Comments1=\n")
                    lay.write("Comments2=\n")
                    lay.write("\n")

                    lay.write("[Montage]\n")
                    lay.write("\n")

                    lay.write("[SampleTimes]\n")
                    lay.write(f"0={first_sample_secs}\n")
                    lay.write("\n")
                        
                    lay.write("[ChannelMap]\n")
                    # Write channel names
                    for c in range(len(elec_name_list)):
                        lay.write(f"{elec_name_list[c]}-Ref={c+1}\n")
                    lay.write("\n")

                    lay.write("[Comments]\n")
                    lay.write("\n")

                print(f"Done writing {new_lay_file_path}")

                # Read the written .laydat file and compare with the original .laydat file
                written_eeg_hdr = mne.io.read_raw_persyst(new_lay_file_path, verbose=False)
                written_eeg_data = written_eeg_hdr.get_data(start=0, stop=None)

                # Calculate the error between the original and written EEG data
                non_zero_samples_sel = eeg_data.flatten()!=0
                err_ls = np.abs((eeg_data.flatten()-written_eeg_data.flatten())[non_zero_samples_sel]/eeg_data.flatten()[non_zero_samples_sel])*100
                errors_df = pd.DataFrame({'FileID':[new_lay_file_path.stem], 'MaxErr%':[np.max(err_ls)], 'MinErr%':[np.mean(err_ls)], 'MedianErr%':[np.median(err_ls)], 'MeanErr%':[np.mean(err_ls)], 'StdErr%':[np.std(err_ls)]})
                this_patient_errors_df = pd.concat([this_patient_errors_df, errors_df])

    return this_patient_errors_df

if __name__ == "__main__":
    
    eeg_data_path = Path("F:/Pediatric_Patients_Simultaneous/")
    patients = {
    'PAT001':2,
    }

    all_errors_df = pd.DataFrame()
    for pat_nr, pat_name in enumerate(patients.keys()):
        pat_id=patients[pat_name]#pat_nr+1
        output_path = Path("F:/Pediatric_Patients_Simultaneous/One_Hour_Converted_Files/") / f"Pat{pat_id:05d}"
        os.makedirs(output_path, exist_ok=True)
        this_pat_eeg_data_path = eeg_data_path / pat_name
        errors_df = write_dat(eeg_data_path=this_pat_eeg_data_path, pat_id=pat_id, output_path=output_path)
        all_errors_df = pd.concat([all_errors_df, errors_df], axis=0)
        all_errors_df.to_csv(output_path / "AllErrors.csv", index=False)
