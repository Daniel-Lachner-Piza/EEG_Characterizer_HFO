import sys
import run_eeg_hfo_characterize_detect as rhd

def create_test_args():
    """Create test arguments for debugging mode."""
    class TestArgs:
        def __init__(self):
            self.dataset_name = "PhysioTest_DLP"
            self.rm_vchann = "yes"
            #self.input_folder = "/home/dlp/Documents/Development/Data/Physio_EEG_Data/"
            self.input_folder = "/work/jacobs_lab/Arash_EEG_Data/"
            self.output_folder = "/home/dlp/Documents/Development/Data/Test-DLP-Output/"
            self.eeg_format = "edf"
            self.montage_type = "sb"
            self.montage_channels = "" #"F3-C3,C3-P3,F4-C4,C4-P4"
            self.power_line_freq = 60
            self.force_characterization = "yes"
            self.force_hfo_detection = "yes"
            self.start_sec = 0.0
            self.end_sec = -1.0
            self.wdw_step_s = 0.1
            self.n_jobs = -1
            self.verbose = "yes"
    
    return TestArgs()

def main() -> None:
    """Main execution function."""

    # Automatically enable test mode when running in debugger or no arguments passed
    test_mode = sys.gettrace() is not None or len(sys.argv) == 1

    if test_mode:
        args = create_test_args()
    else:
        parser = rhd.create_argument_parser()
        args = parser.parse_args()

    cfg = rhd.Characterization_Config(args)
    data_extractor = rhd.InputDataExtractor(cfg)
    data_extractor.get_files_to_process()

    # Display and log files to process
    data_extractor.display_files_to_process()


if __name__ == "__main__":
    main()
