from glob import glob
from pathlib import Path
from natsort import natsorted
# from datetime import datetime, timedelta
import subprocess, os, time

# To use after registration
# Different folder structure at the moment
""" V2 for registered outputs"""


# dhcp_datadir = "/Users/arnaud/Downloads/sourcedata"
# bobs_datadir = "/Users/arnaud/Downloads/bobs_mri"
dhcp_reg_datadir = "./mri_register/dhcp" 

# dhcp_T1ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T1w.*"))
# dhcp_T2ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T2w.*"))
# bobs_T1ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T1w.*"))
# bobs_T2ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T2w.*"))

dhcp_reg_all = natsorted(glob(f"{dhcp_reg_datadir}/*.nii.gz"))


def main(dataset = "bobs", test = False):
    if dataset == 'bobs':
        input_data = bobs_T1ws
        outputdir = "./mri_segs/bobs"
        print(f"Total bobs T1w :{len(input_data)}")
    elif dataset == 'dhcp':
        input_data = dhcp_reg_all
        outputdir = "./mri_segs/dhcp/noparc"
        print(f"Total dhcp T1w :{len(input_data)}")

    csvdir = f"{outputdir}/csv"
    logdir = f"{csvdir}/synthseged_files.txt"

    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(csvdir, exist_ok=True)
    Path(logdir).touch(exist_ok=True)

    with open(logdir, "r") as f:
        processed_files = set(f.read().splitlines())

    print(f"Already started files : {len(processed_files)}/{len(input_data)}.")
    print(f"{processed_files}")
    before = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Already Completed files {len(processed_files)}/{len(input_data)}. \n")
    # print(f"{before}")

    # Initialize timing statistics
    total_start_time = time.time()
    processed_count = 0

    if not test:
        with open(logdir, "a") as logfile:
            for file in input_data[:100]:
                # folder structure check
                parts = file.split("/")
                subject_id = parts[-4] # sub-XXXX
                session_id = parts[-3] # ses-XX
                img = parts[-1]

                # if after seg
                name = img[:-10]
                output_filename = f"{name}_synthseg_robust_cpu_frodo.nii.gz"

                # if source
                # output_filename = f"{subject_id}_{session_id}_synthseg_robust_cpu_mac.nii.gz"

                output_path = f"{outputdir}/{output_filename}"
                csv_path = f"{csvdir}/{output_filename}_vol.csv"
                qc_path = f"{csvdir}/{output_filename}_qc.csv"

                # check if already processed and csvs present
                if f"{output_filename}" in processed_files and os.path.exists(output_path):
                    pass
                    if os.path.exists(csv_path) and os.path.exists(qc_path):
                        print(f"Skipping already processed and logged : {output_filename} \n")
                        continue
                    else:
                        print(f"Processed but not logged : {output_filename} \n")

                # synthseg
                try:
                    print(f"Processing and logging {output_filename} : ")
                    print(f"{file}")
                    subprocess.run(["mri_synthseg", "--i", file, "--o", output_path,
                                    "--vol", csv_path, "--qc", qc_path, #"--parc",
                                    "--robust", "--cpu", "--threads", "8"], check=True)

                    logfile.write(f"{output_filename}\n")
                    print(f"Saved synthseg to {os.path.abspath(output_path)}. \n")

                except KeyboardInterrupt:
                    print("\nProcessing interrupted. Progress has been saved.")
                    break

                except subprocess.CalledProcessError as e:
                    print(f"Error processing {output_filename}: {str(e)}")

        after = natsorted(glob(f"{outputdir}/*.nii.gz"))
        print(f"Newly completed files {len(after) - len(before)}")

    if test: print(input_data)

if __name__ == '__main__':
    main(dataset='dhcp', test=False)
