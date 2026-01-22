# data/utils.py

import shlex
import subprocess
import sys
import matplotlib.pyplot as plt
import nibabel as nib
import os
import platform

def visualize_nifti_depth(file_path, slices=3, save=False):
    """
    Visualize the depth slices of a NIfTI file along all three axes.

    Parameters:
    file_path (str): Path to the NIfTI file.
    slices (int): Number of slices to visualize along each axis.
    save (bool): Whether to save the figure.
    """
    # Load 
    img = nib.load(file_path)
    data = img.get_fdata()
    if data.ndim != 3:
        raise ValueError("The NIfTI file must be 3-dimensional.")

    # middle slices + slice size
    mid_slice_x = data.shape[0] // 2
    mid_slice_y = data.shape[1] // 2
    mid_slice_z = data.shape[2] // 2
    fifth_x = data.shape[0] // 5
    fifth_y = data.shape[1] // 5
    fifth_z = data.shape[2] // 5

    slices_x = [data[mid_slice_x - fifth_x, :, :], data[mid_slice_x, :, :], data[mid_slice_x + fifth_x, :, :]]
    slices_y = [data[:, mid_slice_y - fifth_y, :], data[:, mid_slice_y, :], data[:, mid_slice_y + fifth_y, :]]
    slices_z = [data[:, :, mid_slice_z - fifth_z], data[:, :, mid_slice_z], data[:, :, mid_slice_z + fifth_z]]

    # figure
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # x-axis plots
    for ax, slice_data in zip(axes[0], slices_x):
        ax.imshow(slice_data.T, cmap='gray', origin='lower')
        ax.axis('off')

    # y-axis plots
    for ax, slice_data in zip(axes[1], slices_y):
        ax.imshow(slice_data.T, cmap='gray', origin='lower')
        ax.axis('off')

    # z-axis plots
    for ax, slice_data in zip(axes[2], slices_z):
        ax.imshow(slice_data.T, cmap='gray', origin='lower')
        ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig('nifti_depth_visualization.png')

    return fig

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
        if is_exe(os.path.join(os.getenv('FREESURFER_HOME'),'bin',program)):
            return os.path.join(os.getenv('FREESURFER_HOME'),'bin',program)
        if is_exe(os.path.join('.',program)):
            return os.path.join('.',program)

    return None

def run_cmd(cmd,err_msg):
    """
    execute the comand
    """
    clist = cmd.split()
    progname=which(clist[0])
    if (progname) is None:
        print('ERROR: '+ clist[0] +' not found in path!')
        sys.exit(1)
    clist[0]=progname
    cmd = ' '.join(clist)
    print('#@# Command: ' + cmd+'\n')

    args = shlex.split(cmd)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr != b'':
        print('ERROR: '+ err_msg)

    return stdout

def is_docker_running():
    """Check if Docker daemon is running."""
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def start_docker():
    """Start Docker daemon based on the OS."""
    system = platform.system()

    try:
        if system == "Linux":
            print("Starting Docker daemon on Linux...")
            subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
        elif system == "Darwin":  # macOS
            print("Starting Docker daemon on macOS...")
            subprocess.run(["open", "-a", "Docker"], check=True)
        elif system == "Windows":
            print("Starting Docker daemon on Windows...")
            subprocess.run(["powershell", "Start-Service", "docker"], check=True, shell=True)
        else:
            print(f"Unsupported OS: {system}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker: {e}")
        sys.exit(1)
