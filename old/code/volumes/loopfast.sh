# v2, using Fastsurfer
# loop through the dhcp dataset 
# currenlty doing only the seg : --seg_only

source=/home/data/dhcp_mri/sourcedata/
for folder in "$source"/sub*/ses*/anat; 
do
    sessid=$(echo "$folder" | grep -oP 'ses-\d+')
    for file in "$folder"/*; 
    do
        # Find T1 files ONLY that don't contain 'skull' in their name   
        if [[ $file == *"T1w"* && $file != *"skull"* ]]; then
            echo "Processing Session ID: $sessid - T1 file: $file"
            # end of pwd only
            endpth=$(echo "$file" | awk -F'/' '{n=NF-1; for(i=n-2;i<=n;i++) printf "%s/", $i; print $NF}')
            # Echo the last 3 parent directories and the file
            # echo "$endpth"

            # docker loop
            docker run --gpus all -v /home/data/dhcp_mri/sourcedata/:/data \
                                  -v /home/arnaud/Downloads/dockerout:/output \
                                  -v /home/arnaud/Desktop:/fs_license \
                                  --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                                  --fs_license /fs_license/license.txt \
                                  --t1 /data/$endpth \
                                  --sid $sessid --sd /output \
                                  --parallel --3T \
                                  --seg_only
        fi
    done
done