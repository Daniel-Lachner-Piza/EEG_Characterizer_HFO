import mne
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Tuple       

class EEG_IO:
    """
    A class to handle EEG data reading, channel selection, and data retrieval.

    Attributes:
    eeg_filepath (str): The path to the EEG file.
    mtg_t (str): The montage type. Supported types are 'sr' (Scalp Referential), 'sb' (Scalp Long Bipolar), 'ir' (Intracranial Referential), 'ib' (Intracranial Bipolar).
    eeg_hdr (mne.io.BaseRaw): The MNE Raw object containing the EEG header information.
    get_ch_names (function): A function to get the channel names based on the montage type.
    get_data (function): A function to get the EEG data based on the montage type.
    filename (str): The filename of the EEG file.
    fs (float): The sampling frequency of the EEG data.
    ch_names (list): The channel names.
    ch_indices (list): The channel indices.
    data (numpy.ndarray): The EEG data.
    units (str): The units of the EEG data.
    time_s (numpy.ndarray): The time samples.
    n_samples (int): The number of samples in the EEG data.

    Methods:
    read_eeg_header(eeg_filepath:str=None)->mne.io.BaseRaw:
        Read the EEG header information from the specified file.
    clean_channel_labels(ch_names:list=None)->list:
        Clean the channel names by removing empty spaces and the '-Ref' string.
    get_scalp_ref_chann_labels()->Tuple[list,list]:
        Get the channel names and indices for Scalp Referential montage type.
    get_scalp_long_bip_chann_labels()->Tuple[list,list]:
        Get the channel names and indices for Scalp Long Bipolar montage type.
    get_ieeg_ref_chann_labels()->Tuple[list,list]:
        Get the channel names and indices for Intracranial Referential montage type.
    get_ieeg_bip_chann_labels()->Tuple[list,list]:
        Get the channel names and indices for Intracranial Bipolar montage type.
    get_referential_data(picks:int|list=None, start:int=0, stop:int=None, plot_ok:bool=False)->np.ndarray:
        Get the EEG data for the specified referential channels.
    get_bipolar_data(picks:int|list=None, start:int=0, stop:int=None, plot_ok:bool=False)->np.ndarray:
        Get the EEG data for the specified bipolar channels.
    plot_sample_wdw_per_ch()->None:
        Plot a sample window of the EEG data for each channel.
    plot_ch_signal(filename:str=None, ch_signal:np.ndarray=None, ch_name:Tuple[str,str]=None, units_str:str=None, time_s:np.ndarray=None, plt_wdw_s:Tuple[int,int]=None)->None:
        Plot the EEG signal for a specific channel within a specified time window.
    get_all_possible_scalp_long_bip_labels()->list:
        Get all possible Scalp Long Bipolar channel labels.
    get_valid_scalp_channel_labels()->list:
        Get the valid Scalp channel labels.
    get_non_eeg_channel_labels()->list:
        Get the non-EEG channel labels.
    """
    def __init__(self, eeg_filepath:str=None, mtg_t:str='ir')->None:
        """
        Initialize EEG data reader and define functions to get channel names and data.

        Parameters:
        eeg_filepath (str): The path to the EEG file.
        mtg_t (str): The montage type. Supported types are 'sr' (Scalp Referential), 'sb' (Scalp Long Bipolar), 'ir' (Intracranial Referential), 'ib' (Intracranial Bipolar).

        Returns:
        None
        """
        self.eeg_hdr = self.read_eeg_header(eeg_filepath)

        # Define functions to link with get_data and get_ch_names functions
        if mtg_t=='sr':
            self.get_ch_names = self.get_scalp_ref_chann_labels
            self.get_data = self.get_referential_data
        elif mtg_t=='sb':
            self.get_ch_names = self.get_scalp_long_bip_chann_labels
            self.get_data = self.get_bipolar_data
        elif mtg_t=='ir':
            self.get_ch_names = self.get_ieeg_ref_chann_labels
            self.get_data = self.get_referential_data
        elif  mtg_t=='ib':
            self.get_ch_names = self.get_ieeg_bip_chann_labels
            self.get_data = self.get_bipolar_data
        else:
            raise ValueError('Invalid montage type. Supported types are sr, sb, ir, ib.')

        self.filename = self.eeg_hdr.filenames[0].parts[-1]
        self.fs = int(self.eeg_hdr.info["sfreq"])
        self.ch_names = self.get_ch_names()[0]
        self.ch_indices = self.get_ch_names()[1]
        self.data = []
        self.units = self.eeg_hdr._orig_units
        self.time_s = self.eeg_hdr.times
        self.n_samples = int(self.eeg_hdr.n_times)
        self.meas_date = self.eeg_hdr.info['meas_date']

        # if mne raw reader doesn't read the original units, set default units to uV
        if len(self.units) == 0:
            self.units = 'uV'

        pass



    def read_eeg_header(self, eeg_filepath:str=None) -> mne.io.BaseRaw:
        """
        Read the EEG header from the specified file path.

        Parameters:
        eeg_filepath (str): The file path of the EEG data file.

        Returns:
        eeg_hdr (mne.io.BaseRaw): The EEG header object.

        The function reads the EEG data file based on the file extension and returns the corresponding EEG header object using the appropriate MNE-Python function.
        """
        from pathlib import Path
        filepath = Path(eeg_filepath)
        file_extension = filepath.suffixes[0].lower()
        eeg_hdr = None
        if file_extension=='.lay':
            eeg_hdr = mne.io.read_raw_persyst(eeg_filepath, verbose='ERROR')
        elif file_extension=='.edf':
            try:
                eeg_hdr = mne.io.read_raw_edf(eeg_filepath, verbose='ERROR')
            except Exception as e:
                print(f"Error reading EDF file with default encoding: {e}")
                print("Trying with 'latin1' encoding...")
                eeg_hdr = mne.io.read_raw_edf(eeg_filepath, verbose='ERROR', encoding='latin1')
        elif file_extension=='.vhdr':
            eeg_hdr = mne.io.read_raw_brainvision(eeg_filepath, verbose='ERROR')

        if eeg_hdr is None:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return eeg_hdr

    
    def clean_channel_labels(self, ch_names:list=None)->list:
        """
        Cleans the channel labels by removing empty spaces and the '-Ref' string, and by removing leading zeros.

        Parameters:
        ch_names (list): A list of channel names to be cleaned.

        Returns:
        list: A list of cleaned channel names.
        """
        for chi, ch_name in enumerate(ch_names):
            # Clean channel names from empty spaces and the '-Ref string'
            ch_name = ch_name.lstrip(" ").replace('-Ref', '')
            ch_name = ch_name.lstrip(" ").replace('EEG ', '')

            # Clean channel names from leading zeros'
            alpha_char_ls = [c for c in ch_name if c.isalpha()]
            alpha_str = ''
            if len(alpha_char_ls) > 0:
                alpha_str = ''.join(alpha_char_ls)

            dig_char_ls = [c for c in ch_name if c.isdigit()]
            dig_str = ''
            if len(dig_char_ls) >0:
                dig_str = str(int(''.join(dig_char_ls)))

            ch_name = alpha_str + dig_str
            ch_names[chi] = ch_name
        return ch_names


    def get_scalp_ref_chann_labels(self) -> Tuple[list, list]:
        """
        Retrieves the labels and indices of the Scalp EEG referential channels.

        Parameters:
        None

        Returns:
        scalp_eeg_ch_labels (list): A list of Scalp EEG channel labels.
        scalp_eeg_chs_indxs (list): A list of Scalp EEG channel indices.

        The function iterates through the channel names in the EEG header, checks if each channel is a valid Scalp EEG channel, and appends the channel label and index to the respective lists. The channel labels are then cleaned using the `clean_channel_labels` method.
        """
        valid_ch_names = [chname.lower() for chname in self.get_valid_scalp_channel_labels()]
        scalp_eeg_chs_indxs = []
        scalp_eeg_ch_labels = []
        for ch_idx, ch_name in enumerate(self.eeg_hdr.ch_names):
            try:
                valid_ch_names.index(ch_name.lower())
            except ValueError:
                print(f"Channel {ch_name} is an invalid Scalp EEG channel label")
                continue
            scalp_eeg_chs_indxs.append(ch_idx)
            scalp_eeg_ch_labels.append(ch_name)
        scalp_eeg_ch_labels = self.clean_channel_labels(scalp_eeg_ch_labels)
        return scalp_eeg_ch_labels, scalp_eeg_chs_indxs


    def get_scalp_long_bip_chann_labels(self) -> Tuple[list, list]:
        """
        Retrieves the labels and indices of the Scalp Longitudinal Bipolar EEG channels.

        Parameters:
        None

        Returns:
        mtg_labels_ls (list): A list of Scalp Long Bipolar EEG channel labels.
        mtg_chs_indices_ls (list): A list of tuples, where each tuple contains the indices of the two channels forming a bipolar montage.

        The function iterates through the channel names in the EEG header, checks if each channel is a valid Scalp Longitudinal Bipolar EEG channel, and appends the channel label and index to the respective lists. The channel labels are then cleaned using the `clean_channel_labels` method.
        """
        
        ch_names = self.clean_channel_labels(self.eeg_hdr.ch_names)
        ch_names_low = [chn.lower() for chn in ch_names]      
        mtg_labels_ls = []
        mtg_chs_indices_ls = []
        for bip_mtg in self.get_all_possible_scalp_long_bip_labels():
            bip_mtg_parts = bip_mtg.split("-")
            bip_mtg_parts = [mtgname.lower() for mtgname in bip_mtg_parts]
            try:
                ch_1_idx = ch_names_low.index(bip_mtg_parts[0])
            except ValueError:
                print(f"Channel {bip_mtg_parts[0]} not found in EEG")
                continue
            try:
                ch_2_idx = ch_names_low.index(bip_mtg_parts[1])
            except ValueError:
                print(f"Channel {bip_mtg_parts[1]} not found in EEG")
                continue
            mtg_labels_ls.append(bip_mtg)
            mtg_chs_indices_ls.append((ch_1_idx, ch_2_idx))
        return mtg_labels_ls, mtg_chs_indices_ls

    
    def get_ieeg_ref_chann_labels(self) -> Tuple[list, list]:
        """
        Retrieves the labels and indices of the intracranial EEG (iEEG) referential channels.

        Parameters:
        None

        Returns:
        ieeg_ch_labels (list): A list of iEEG channel labels.
        ieeg_chs_indxs (list): A list of iEEG channel indices.

        The function iterates through the channel names in the EEG header, checks if each channel is a valid iEEG channel, and appends the channel label and index to the respective lists. The channel labels are then cleaned using the `clean_channel_labels` method.
        """
        valid_scalp_ch_names = [chname.lower() for chname in self.get_valid_scalp_channel_labels()]
        non_eeg_ch_names = [chname.lower() for chname in self.get_non_eeg_channel_labels()]
        ieeg_ch_labels = []
        ieeg_chs_indxs = []
        for ch_idx, ch_name in enumerate(self.eeg_hdr.ch_names):
            ch_name_hw_group = ''.join([c for c in ch_name if not c.isdigit()])
            try:
                valid_scalp_ch_names.index(ch_name.lower())
            except ValueError:
                try:
                    non_eeg_ch_names.index(ch_name_hw_group.lower())
                except ValueError:
                    ieeg_ch_labels.append(ch_name)
                    ieeg_chs_indxs.append(ch_idx)
        ieeg_ch_labels = self.clean_channel_labels(ieeg_ch_labels)
        return ieeg_ch_labels, ieeg_chs_indxs


    def get_ieeg_bip_chann_labels(self) -> Tuple[list, list]:
        """
        Retrieves the labels and indices of the intracranial EEG (iEEG) bipolar channels.

        Parameters:
        None

        Returns:
        mtg_labels_ls (list): A list of iEEG bipolar channel labels.
        mtg_chs_indices_ls (list): A list of tuples, where each tuple contains the indices of the two channels forming a bipolar montage.

        The function first retrieves the labels and indices of the iEEG reference channels using the `get_ieeg_ref_chann_labels` method.
        Then, it iterates through the iEEG reference channels to find pairs of channels that form a valid bipolar montage.
        The bipolar channel labels and their corresponding indices are stored in the `mtg_labels_ls` and `mtg_chs_indices_ls` lists, respectively.
        Finally, the channel labels are cleaned using the `clean_channel_labels` method before being returned.
        """
        ieeg_ch_labels, ieeg_chs_indxs = self.get_ieeg_ref_chann_labels()
        mtg_labels_ls = []
        mtg_chs_indices_ls = []

        for chname_a, chidx_a in zip(ieeg_ch_labels, ieeg_chs_indxs):
            hw_group_a = ''.join([c for c in chname_a if not c.isdigit()])
            try:
                contact_nr_a = int(''.join([c for c in chname_a if c.isdigit()]))
            except ValueError:
                print(f"Invalid channel name format: {chname_a}")
                continue

            for chname_b, chidx_b in zip(ieeg_ch_labels, ieeg_chs_indxs):
                hw_group_b = ''.join([c for c in chname_b if not c.isdigit()])
                try:
                    contact_nr_b = int(''.join([c for c in chname_b if c.isdigit()]))
                except ValueError:
                    continue

                if (hw_group_a==hw_group_b) and (contact_nr_b-contact_nr_a==1):
                    mtg_name = f"{chname_a}-{chname_b}"
                    mtg_labels_ls.append(mtg_name)
                    mtg_chs_indices_ls.append((chidx_a, chidx_b))
        return mtg_labels_ls, mtg_chs_indices_ls


    def remove_natus_virtual_channels(self) -> Tuple[list, list]:
        
        accepted_channs = ["c3", "c4", "cz"]
        clean_ch_names = []
        clean_ch_indices = []
        for idx in range(len(self.ch_names)):
            mtg_name = self.ch_names[idx]
            mtg_idxs = self.ch_indices[idx]
            # Bipolar montage
            if len(mtg_name.split('-'))>1:
                mtg_ch_a = mtg_name.split('-')[0]
                mtg_ch_b = mtg_name.split('-')[1]

                hw_group_a = ''.join([c for c in mtg_ch_a if not c.isdigit()])
                hw_group_b = ''.join([c for c in mtg_ch_b if not c.isdigit()])

                if (hw_group_a=="C" or hw_group_b=="C") or (hw_group_a=="DC" or hw_group_b=="DC"):
                    if (mtg_ch_a.lower() not in accepted_channs and mtg_ch_b.lower() not in accepted_channs):
                        print(f"Exclude channel {mtg_name}")
                        continue
                clean_ch_names.append(mtg_name)
                clean_ch_indices.append(mtg_idxs)
            else:
            # Referential montage
                hw_group = ''.join([c for c in mtg_name if not c.isdigit()])
                contact_nr = ''.join([c for c in mtg_name if c.isdigit()])
                if (hw_group=="C" or hw_group=="DC" or len(contact_nr)==0):
                    if mtg_name.lower() not in accepted_channs:
                        print(f"Exclude channel {mtg_name}")
                        continue
                clean_ch_names.append(mtg_name)
                clean_ch_indices.append(mtg_idxs)
            pass
        
        self.ch_names = clean_ch_names
        self.ch_indices = clean_ch_indices

        pass
    
    def get_referential_data(self, picks:int|list=None, start:int=0, stop:int=None, plot_ok:bool=False)->np.ndarray:
        """
        Retrieves the EEG data for the specified reference channels.

        Parameters:
        picks (int|list): The indices of the channels to be picked. If None, all channels are picked.
        start (int): The starting sample index.
        stop (int): The stopping sample index.
        plot_ok (bool): A flag indicating whether to plot the data.

        Returns:
        self.data (numpy.ndarray): The EEG data for the specified reference channels.
        """
        mtg_picks = self.ch_indices
        if picks is not None:
            mtg_picks = self.ch_indices[picks]
        self.data = self.eeg_hdr.get_data(picks=mtg_picks, start=start, stop=stop)
        return self.data


    
    def get_bipolar_data(self, picks:int|list=None, start:int=0, stop:int=None, plot_ok:bool=False)->np.ndarray:
        """
        Retrieves the EEG data for the specified bipolar channels.

        Parameters:
        picks (int|list): The indices of the channels to be picked. If None, all channels are picked.
        start (int): The starting sample index.
        stop (int): The stopping sample index.
        plot_ok (bool): A flag indicating whether to plot the data.

        Returns:
        self.data (numpy.ndarray): The EEG data for the specified bipolar channels.
        """
        # Get data
        mtg_picks_a = [chi[0] for chi in self.ch_indices]
        mtg_picks_b = [chi[1] for chi in self.ch_indices]
        assert len(mtg_picks_a) == len(mtg_picks_b), "Bipolar montage channel indices are not equal"

        if stop is None:
            stop = self.n_samples
        if start < 0:
            start = 0
        if stop > self.n_samples:
            stop = self.n_samples
        if start > stop:
            raise ValueError("Start sample is greater than stop sample")
        if start == stop:
            raise ValueError("Start sample is equal to stop sample")
        

        if picks is not None:
            mtg_picks_a = mtg_picks_a[picks]
            mtg_picks_b = mtg_picks_b[picks]
        
        if isinstance(mtg_picks_a, int):
            sig_a = self.eeg_hdr.get_data(picks=mtg_picks_a, start=start, stop=stop)
            sig_b = self.eeg_hdr.get_data(picks=mtg_picks_b, start=start, stop=stop)
            self.data = sig_a - sig_b
            self.data = self.data.flatten()
            pass
        else:
            self.data = np.zeros((len(mtg_picks_a), stop-start))
            for i in range(len(mtg_picks_a)):
                #print(f"Channel {self.ch_names[i]}")
                sig_a = self.eeg_hdr.get_data(picks=mtg_picks_a[i], start=start, stop=stop)
                sig_b = self.eeg_hdr.get_data(picks=mtg_picks_b[i], start=start, stop=stop)
                self.data[i] = sig_a - sig_b
                pass
        return self.data


    
    def plot_sample_wdw_per_ch(self) -> None:
        """
        Plots a sample window of the EEG signal for each channel.

        Parameters:
        None

        Returns:
        None

        The function iterates through each channel in the EEG data, selects a sample window, and calls the `plot_ch_signal` method to plot the signal for that channel.
        """
        for chidx, ch_name in enumerate(self.ch_names):
            ch_signal = self.data[chidx]
            plt_wdw_s = (10, 20)
            self.plot_ch_signal(
                self.filename,
                ch_signal,
                ch_name,
                self.units,
                self.time_s,
                plt_wdw_s,
            )

    def plot_ch_signal(
        self,
        filename: str = None,
        ch_signal: np.ndarray = None,
        ch_name: Tuple[str, str] = None,
        units_str: str = None,
        time_s: np.ndarray = None,
        plt_wdw_s: Tuple[int, int] = None,
    ) -> None:
        """
        Plots the EEG signal for a specific channel within a given time window.

        Parameters:
        filename (str): The name of the EEG file.
        ch_signal (np.ndarray): The EEG signal data for the specific channel.
        ch_name (Tuple[str, str]): The name of the specific channel.
        units_str (str): The units of the EEG signal data.
        time_s (np.ndarray): The time samples corresponding to the EEG signal data.
        plt_wdw_s (Tuple[int, int]): The start and end time of the time window for plotting.

        Returns:
        None
        """
        sample_sel = (time_s >= plt_wdw_s[0]) & (time_s <= plt_wdw_s[1])
        signal_to_plot = ch_signal[sample_sel] * -1
        time_to_plot = time_s[sample_sel]

        # Plot signal
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"{filename}\n EEG Montage Signal: {ch_name}")

        sig_lw = 0.5
        plt.plot(time_to_plot, signal_to_plot, "-", color="black", linewidth=sig_lw)
        plt.title(ch_name)
        plt.xlim(min(time_to_plot), max(time_to_plot))
        plt.ylim(min(signal_to_plot), max(signal_to_plot))
        plt.ylabel(f"Amplitude ({units_str})")
        plt.xlabel("Time (s)")
        plt.legend([ch_name], loc="upper right")

        #plt.waitforbuttonpress()
        plt.show()
        plt.pause(0.5)
        plt.close()


    def get_all_possible_scalp_long_bip_labels(self) -> list:
        """
        Returns a list of all possible scalp longitudinal bipolar montage labels.

        Parameters:
        None

        Returns:
        scalp_long_bip (list): A list of strings representing the scalp longitudinal bipolar montage labels.
        """
        scalp_long_bip = [
            "Fp1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "F7-T3",
            "T3-T5",
            "T5-O1",
            "Fp2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "F8-T4",
            "T4-T6",
            "T6-O2",
            "Fp1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "Fp2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "FZ-CZ",
            "CZ-PZ",
        ]

        return scalp_long_bip


    def get_valid_scalp_channel_labels(self) -> list:
        """
        Returns a list of valid scalp channel labels.

        Parameters:
        None

        Returns:
        scalp_labels (list): A list of strings representing the valid scalp channel labels.
        """
        scalp_labels = [
            "A1",
            "A2",
            "C3",
            "C4",
            "CZ",
            "F3",
            "F4",
            "F7",
            "F8",
            "Fp1",
            "Fp2",
            "FpZ",
            "FZ",
            "M1",
            "M2",
            "O1",
            "O2",
            "Oz",
            "P3",
            "P4",
            "P7",
            "P8",
            "PZ",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "FP1",
            "F7",
            "T3",
            "T5",
            "F3",
            "C3",
            "P3",
            "O1",
            "FP2",
            "F8",
            "T4",
            "T6",
            "F4",
            "C4",
            "P4",
            "O2",
            "FZ",
            "CZ",
            "A1",
            "A2",
            "TP9",
            "FT9",
            "TP10",
            "FT10",
            "P9",
            "P10",
            "Fz",
            "Cz",
            "Pz",
        ]
        return scalp_labels


    def get_non_eeg_channel_labels(self) -> list:
        """
        Retrieves a list of non-EEG channel labels.

        Parameters:
        None

        Returns:
        non_eeg_labels (list): A list of strings representing the non-EEG channel labels.
        """
        non_eeg_labels = ["ECG", "EKG", "EKGR", "EKGL", "EOG", "EMG", "EOG", "TRIG", "OSAT", "PLETH", "EVENT"]

        return non_eeg_labels

