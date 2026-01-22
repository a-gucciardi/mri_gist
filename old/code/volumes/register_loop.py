import ants
from glob import glob
from pathlib import Path
from natsort import natsorted
import subprocess, os, time
import tqdm


dhcp_datadir = "/home/data/dhcp_mri/sourcedata_orig" # dataset without skullstrips
# dhcp_datadir = "/Users/arnaud/Downloads/sourcedata_orig"
# bobs_datadir = "/Users/arnaud/Downloads/bobs_mri"

dhcp_T1ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T1w.*"))
dhcp_T2ws = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*T2w.*"))
dhcp_all = natsorted(glob(f"{dhcp_datadir}/sub*/ses*/anat/*"))
# bobs_T1ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T1w.*"))
# bobs_T2ws = natsorted(glob(f"{bobs_datadir}/sub*/ses*/anat/*T2w.*"))

def main(dataset = "dhcp", template = "1", test = False):
    if dataset == 'bobs':
        input_data = bobs_T1ws
        outputdir = "./mri_segs/bobs"
        print(f"Total bobs T1w :{len(input_data)}")
    elif dataset == 'dhcp':
        input_data = dhcp_all
        outputdir = "./mri_register/dhcp"
        print(f"Total dhcp :{len(input_data)}")

    os.makedirs(outputdir, exist_ok=True)

    if template == "1":
        template_file = ants.image_read("./registre/Akiyama_6Month_1_5T.nii.gz")
    elif template == "3":
        template_file = ants.image_read("./registre/Akiyama_6Month_3T.nii.gz")

    # logging
    logdir = f"{outputdir}/register_log.txt"
    Path(logdir).touch(exist_ok=True)

    with open(logdir, "r") as f:
        processed_files = set(f.read().splitlines())

    print(f"Already started files : {len(processed_files)}/{len(input_data)}.")
    print(f"{processed_files}")
    before = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Already Completed files {len(processed_files)}/{len(input_data)}. \n")

    if not test:
        with open(logdir, "a") as logfile:
            for file in tqdm.tqdm(input_data):
                # folder structure check
                parts = file.split("/")
                subject_id = parts[-4] # sub-XXXX
                session_id = parts[-3] # ses-XX
                img = parts[-1]

                output_filename = f"{img[:-7]}_MNI_{template}T.nii.gz"
                output_path = f"{outputdir}/{output_filename}"
                # print(output_filename)

                # check if already processed
                if f"{img[:-7]}" in processed_files and os.path.exists(output_path):
                    print(f"Skipping already processed and logged : {subject_id} {session_id} \n")
                    continue

                # ants registration
                try:
                    print(f"Processing and logging {subject_id} {session_id} : ")
                    print(f"{file}")
                    input_image = ants.image_read(file)
                    # Perform registration with SyN algorithm
                    transformed_image = ants.registration(fixed=template_file, moving=input_image, type_of_transform='SyN')
                    ants.image_write(transformed_image['warpedmovout'], f"{output_path}")

                    logfile.write(f"{img[:-7]}\n")
                    print(f"Saved registered image to {os.path.abspath(output_path)}. \n")

                except KeyboardInterrupt:
                    print("\nProcessing interrupted. Progress has been saved.")
                    break

                except Exception as e:
                    print(f"Error processing {subject_id} {session_id}: {str(e)}")
    after = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Newly completed files {len(after) - len(before)}")

if __name__ == '__main__':
    main(dataset="dhcp", test=False, template="1")
