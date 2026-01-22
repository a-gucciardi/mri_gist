from glob import glob
from pathlib import Path
from natsort import natsorted
# from datetime import datetime, timedelta
import subprocess, os, time

dhcp_datadir = "/Users/arnaud/Downloads/sourcedata"
bobs_datadir = "/Users/arnaud/Downloads/bobs_mri"
dhcp_reg_datadir = "./mri_register/dhcp" 

dhcp_T1ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T1w.*"))
dhcp_T2ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T2w.*"))
bobs_T1ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T1w.*"))
bobs_T2ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T2w.*"))

dhcp_reg_all = natsorted(glob(f"{dhcp_reg_datadir}/*.nii.gz"))
# folder structure is always[...]/sub-ID/ses-ID/anat/file

# print(f"Total dhcp T1w :{len(dhcp_T1ws)}")
# print(f"Total dhcp T2w :{len(dhcp_T2ws)}")
# print(f"Total bobs T1w :{len(bobs_T1ws)}")
# print(f"Total bobs T2w :{len(bobs_T2ws)}")


def main(dataset = "bobs", test = False):
    if dataset == 'bobs':
        input_data = bobs_T1ws
        outputdir = "./mri_segs/bobs"
        print(f"Total bobs T1w :{len(input_data)}")
    elif dataset == 'dhcp':
        input_data = dhcp_reg_all
        outputdir = "./mri_segs/dhcp"
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
    before = natsorted(glob(f"{outputdir}/*"))
    print(f"Already Completed files {len(processed_files)}/{len(input_data)}. \n")
    # print(f"{before}")

    # Initialize timing statistics
    total_start_time = time.time()
    processed_count = 0

    if not test:
        with open(logdir, "a") as logfile:
            for file in input_data[:1]:
                # folder structure check
                parts = file.split("/")
                subject_id = parts[-4] # sub-XXXX
                session_id = parts[-3] # ses-XX
                img = parts[-1]

                output_filename = f"{subject_id}_{session_id}_synthseg_robust_parc_cpu_frodo.nii.gz"
                output_path = f"{outputdir}/{output_filename}"
                csv_path = f"{csvdir}/{subject_id}_{session_id}_parc_vol.csv"
                qc_path = f"{csvdir}/{subject_id}_{session_id}_parc_qc.csv"

                # check if already processed and csvs present
                if f"{subject_id}_{session_id}" in processed_files and os.path.exists(output_path):
                    pass
                    if os.path.exists(csv_path) and os.path.exists(qc_path):
                        print(f"Skipping already processed and logged : {subject_id} {session_id} \n")
                        continue
                    else:
                        print(f"Processed but not logged : {subject_id} {session_id} \n")

                # synthseg
                try:
                    print(f"Processing and logging {subject_id} {session_id} : ")
                    print(f"{file}")
                    subprocess.run(["mri_synthseg", "--i", file, "--o", output_path,
                                    "--vol", csv_path, "--qc", qc_path, "--parc",
                                    "--robust", "--cpu", "--threads", "8"], check=True)

                    logfile.write(f"{subject_id}_{session_id}\n")
                    print(f"Saved synthseg to {os.path.abspath(output_path)}. \n")

                except KeyboardInterrupt:
                    print("\nProcessing interrupted. Progress has been saved.")
                    break

                except subprocess.CalledProcessError as e:
                    print(f"Error processing {subject_id} {session_id}: {str(e)}")

        after = natsorted(glob(f"{outputdir}/*"))
        print(f"Newly completed files {len(after) - len(before)}")

if __name__ == '__main__':
    main(dataset='dhcp', test=False)
