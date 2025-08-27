import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from typing import List, Dict, Union, Tuple
from collections import defaultdict


class MontageCreator:
    def __init__(self, eeg_data: mne.io.BaseRaw = None) -> None:

        self.eeg_data = eeg_data

    def get_all_possible_scalp_long_bip_labels(self):
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

    def get_scalp_channel_labels(self):
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

    def get_non_eeg_channel_labels(self):
        non_eeg_labels = ["ECG", "EKG", "EOG", "EMG", "EOG"]
        return non_eeg_labels

    def scalp_long_bip_labels(self):
        print(self.eeg_data)
        print(self.eeg_data.info)

        ch_names = self.eeg_data.ch_names
        ch_names_low = [chn.lower() for chn in ch_names]
        mtg_labels_ls = []
        
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
            
        return mtg_labels_ls

    def scalp_long_bip_data(self, picks=None, start=0, stop=None, plot_ok: bool = False):
        print(self.eeg_data)
        print(self.eeg_data.info)

        data = self.eeg_data.get_data(picks=picks, start=start, stop=stop)
        time_secs = np.array(self.eeg_data.times)
        ch_names = self.eeg_data.ch_names
        ch_names_low = [chn.lower() for chn in ch_names]
        units = self.eeg_data._orig_units
        fs = self.eeg_data.info["sfreq"]
        filename = self.eeg_data.filenames[0].split(os.path.sep)[-1]

        # read_raw_persyst doesn't read the original units, set default units to uV
        if len(units) == 0:
            units = defaultdict(list)
            for chname in ch_names:
                units[chname] = "uV"

        scalp_mtg_data = {
            "filename": filename,
            "fs": fs,
            "mtg_labels": np.array([]),
            "data": np.array([]),
            "n_samples": data.shape[1],
            "units": units[ch_names[0]],
            "time_s": time_secs,
        }

        for bip_mtg in self.get_all_possible_scalp_long_bip_labels():
            bip_mtg_parts = bip_mtg.split("-")
            bip_mtg_parts = [mtgname.lower() for mtgname in bip_mtg_parts]

            try:
                ch_1_idx = ch_names_low.index(bip_mtg_parts[0])
            except ValueError:
                #print(f"Channel {bip_mtg_parts[0]} not found in EEG")
                continue

            try:
                ch_2_idx = ch_names_low.index(bip_mtg_parts[1])
            except ValueError:
                #print(f"Channel {bip_mtg_parts[1]} not found in EEG")
                continue

            mtg_signal = np.array(data[ch_1_idx] - data[ch_2_idx]) * -1
            if len(scalp_mtg_data["data"]) == 0:
                scalp_mtg_data["mtg_labels"] = bip_mtg
                scalp_mtg_data["data"] = mtg_signal
            else:
                scalp_mtg_data["mtg_labels"] = np.append(
                    scalp_mtg_data["mtg_labels"], bip_mtg
                )
                scalp_mtg_data["data"] = np.vstack((scalp_mtg_data["data"], mtg_signal))

            if plot_ok:
                ref_eeg_signals = (data[ch_1_idx], data[ch_2_idx])
                ref_eeg_ch_names = (bip_mtg.split("-")[0], bip_mtg.split("-")[1])
                plt_wdw_s = (10, 20)
                self.plot_montage(
                    filename,
                    ref_eeg_signals,
                    ref_eeg_ch_names,
                    time_secs,
                    plt_wdw_s,
                )

        return scalp_mtg_data
    def get_intracranial_ref_mtg_labels(self):
        eeg_ref_ch_names = self.eeg_data.ch_names
        referential_mtg_labels = defaultdict(list)
        referential_mtg_labels["montageNr"] = list(range(1,len(eeg_ref_ch_names)+1))
        referential_mtg_labels["montageName"] = list(eeg_ref_ch_names)
        return pd.DataFrame(referential_mtg_labels)

    def get_intracranial_bipolar_montage_labels(self):

        unipolar_labels = self.eeg_data.ch_names
        scalp_channels = [ch.lower() for ch in self.get_scalp_channel_labels()]
        non_eeg_channels = [ch.lower() for ch in self.get_non_eeg_channel_labels()]
        referential_contacts = []
        bipolar_montages_info = defaultdict(list)

        # Get Referential contacts
        for chidx, ch_label in enumerate(unipolar_labels):

            # print(ch_label)

            temp = re.compile("([a-zA-Z]+)([0-9]+)")

            if temp.match(ch_label) != None:
                res = temp.match(ch_label).groups()

                electrode_name = res[0]
                contact_nr = res[1]

                natus_virtual_ch = electrode_name == "C" and len(contact_nr) > 0

                if electrode_name.lower() not in non_eeg_channels:
                    if not natus_virtual_ch:
                        if ch_label.lower() not in scalp_channels:
                            referential_contacts.append(
                                (electrode_name, int(contact_nr), chidx)
                            )
                        else:
                            print("Scalp Channels mixed with Intracranial: ", ch_label)
                    else:
                        print("Natus Virtual Channel:", ch_label)
                else:
                    print("Non EEG Channels in data: ", ch_label)

        # Get Bipolar Montages
        montage_nr = 1
        for upi, ref_contact_a in enumerate(referential_contacts):
            ref_contact_a = referential_contacts[upi]
            first_electrode_name = ref_contact_a[0]
            first_contact_nr = ref_contact_a[1]
            first_contact_global_idx = ref_contact_a[2]

            for supi, ref_contact_b in enumerate(referential_contacts):
                if upi != supi:
                    second_electrode_name = ref_contact_b[0]
                    second_contact_nr = ref_contact_b[1]
                    second_contact_global_idx = ref_contact_b[2]

                    montage_name = (
                        f"{first_electrode_name}{first_contact_nr}-"
                        f"{second_electrode_name}{second_contact_nr}"
                    )
                    montage_mossdet_nr = montage_nr

                    if (
                        first_electrode_name == second_electrode_name
                        and second_contact_nr - first_contact_nr == 1
                    ):

                        bipolar_montages_info["firstElectrodeName"].append(
                            first_electrode_name
                        )
                        bipolar_montages_info["firstContactNr"].append(first_contact_nr)
                        bipolar_montages_info["firstContactGlobalIdx"].append(
                            first_contact_global_idx
                        )
                        bipolar_montages_info["secondElectrodeName"].append(
                            second_electrode_name
                        )
                        bipolar_montages_info["secondContactNr"].append(
                            second_contact_nr
                        )
                        bipolar_montages_info["secondContactGlobalIdx"].append(
                            second_contact_global_idx
                        )
                        bipolar_montages_info["montageName"].append(montage_name)
                        bipolar_montages_info["montageChNr"].append(
                            montage_mossdet_nr
                        )
                        montage_nr += 1

        return pd.DataFrame(bipolar_montages_info)

    def intracranial_bipolar(self, start:int=0, stop:int=None, plot_ok: bool = False):
        print(self.eeg_data)
        print(self.eeg_data.info)

        ieeg_mtg_info = self.get_intracranial_bipolar_montage_labels()

        data = self.eeg_data.get_data(start=start, stop=stop) # start=0, stop=None
        time_secs = np.array(self.eeg_data.times)
        ch_names = self.eeg_data.ch_names
        ch_names_low = [chn.lower() for chn in ch_names]
        units = self.eeg_data._orig_units
        fs = self.eeg_data.info["sfreq"]
        filename = self.eeg_data.filenames[0].split(os.path.sep)[-1]

        # read_raw_persyst doesn't read the original units, set default units to uV
        if len(units) == 0:
            units = defaultdict(list)
            for chname in ch_names:
                units[chname] = "uV"

        nr_bip_mtgs = len(ieeg_mtg_info.montageName)
        n_samples = data.shape[1]
        ieeg_mtg_data = {
            "filename": filename,
            "fs": fs,
            "mtg_labels": ["mtg_name"]*nr_bip_mtgs,
            "data": np.zeros((nr_bip_mtgs, n_samples)),
            "n_samples": n_samples,
            "units": units[ch_names[0]],
            "time_s": time_secs,
        }

        bip_mtg_idx = 0
        for ieeg_mtg_idx, ieeg_mtg_name in enumerate(ieeg_mtg_info.montageName):

            ch_1_idx = ieeg_mtg_info.firstContactGlobalIdx[ieeg_mtg_idx]
            ch_2_idx = ieeg_mtg_info.secondContactGlobalIdx[ieeg_mtg_idx]
            mtg_signal = np.array(data[ch_1_idx] - data[ch_2_idx]) * -1
            ieeg_mtg_data["mtg_labels"][bip_mtg_idx] = ieeg_mtg_name
            ieeg_mtg_data["data"][bip_mtg_idx,:] = mtg_signal
            bip_mtg_idx += 1
            # if len(ieeg_mtg_data["data"]) == 0:
            #     ieeg_mtg_data["mtg_labels"] = ieeg_mtg_name
            #     ieeg_mtg_data["data"] = mtg_signal
            # else:
            #     ieeg_mtg_data["mtg_labels"] = np.append(
            #         ieeg_mtg_data["mtg_labels"], ieeg_mtg_name
            #     )
            #     ieeg_mtg_data["data"] = np.vstack((ieeg_mtg_data["data"], mtg_signal))

            if plot_ok:
                ref_eeg_signals = (data[ch_1_idx], data[ch_2_idx])
                ref_eeg_ch_names = (
                    ieeg_mtg_name.split("-")[0],
                    ieeg_mtg_name.split("-")[1],
                )
                plt_wdw_s = (10, 20)
                self.plot_montage(
                    filename,
                    ref_eeg_signals,
                    ref_eeg_ch_names,
                    time_secs,
                    plt_wdw_s,
                )

        return ieeg_mtg_data

    def plot_montage(
        self,
        filename: str = None,
        ref_eeg_signals: Tuple[np.ndarray, np.ndarray] = None,
        ref_eeg_ch_names: Tuple[str, str] = None,
        time_s: np.ndarray = None,
        plt_wdw_s: Tuple[int, int] = None,
    ):
        sample_sel = (time_s >= plt_wdw_s[0]) & (time_s <= plt_wdw_s[1])

        # Plot signals
        left_color = [0, 0.45, 0.74]
        right_color = [0.93, 0.7, 0.13]

        fig, axs = plt.subplots(1, 3, figsize=(16, 9))
        fig.suptitle(f"{filename}\Referential vs Scalp_Long_Bipolar Montage")

        sig_lw = 0.5

        # Channel 1
        plt_ax = axs[0]
        time_to_plot = time_s[sample_sel]
        signal_to_plot = ref_eeg_signals[0][sample_sel] * -1
        ch_to_plot_name = ref_eeg_ch_names[0]
        units_str = "uV"

        plt_ax.plot(time_to_plot, signal_to_plot, "-", color="black", linewidth=sig_lw)
        plt_ax.set_title(ch_to_plot_name)
        plt_ax.set_xlim(min(time_to_plot), max(time_to_plot))
        plt_ax.set_ylim(min(signal_to_plot), max(signal_to_plot))
        plt_ax.set_ylabel(f"Amplitude ({units_str})")
        plt_ax.set_xlabel("Time (s)")
        plt_ax.legend(["Channel 1"], loc="upper right")

        # Channel 2
        plt_ax = axs[1]
        signal_to_plot = ref_eeg_signals[1][sample_sel] * -1
        ch_to_plot_name = ref_eeg_ch_names[1]

        plt_ax.plot(time_to_plot, signal_to_plot, "-", color="black", linewidth=sig_lw)
        plt_ax.set_title(ch_to_plot_name)
        plt_ax.set_xlim(min(time_to_plot), max(time_to_plot))
        plt_ax.set_ylim(min(signal_to_plot), max(signal_to_plot))
        plt_ax.set_ylabel(f"Amplitude ({units_str})")
        plt_ax.set_xlabel("Time (s)")
        plt_ax.legend(["Channel 2"], loc="upper right")

        # Montage
        plt_ax = axs[2]
        signal_to_plot = ref_eeg_signals[0][sample_sel] - ref_eeg_signals[1][sample_sel]
        signal_to_plot *= -1
        ch_to_plot_name = f"{ref_eeg_ch_names[0]} - {ref_eeg_ch_names[1]}"

        plt_ax.plot(time_to_plot, signal_to_plot, "-", color="black", linewidth=sig_lw)
        plt_ax.set_title(ch_to_plot_name)
        plt_ax.set_xlim(min(time_to_plot), max(time_to_plot))
        plt_ax.set_ylim(min(signal_to_plot), max(signal_to_plot))
        plt_ax.set_ylabel(f"Amplitude ({units_str})")
        plt_ax.set_xlabel("Time (s)")
        plt_ax.legend([ch_to_plot_name], loc="upper right")

        plt.draw()
        plt.show()
        plt.close()

        pass
