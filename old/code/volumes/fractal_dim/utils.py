import nibabel as nib
import numpy as np
import math, random
import sklearn.metrics as skl

def fractal_analysis(nib_img, verbose=False):
    ### NIFTI IMAGE LOADING used with path###
    # image_path = volume_path
    # img = nib.load(image_path)
    ### NIFTI IMAGE LOADING used with nibabel img directly###
    img = nib_img
    nii_header = img.header
    imageloaded = img.get_fdata()
    imageloaded.shape
    ### CHECK THE IMAGE ISOTROPY ###
    voxels_size = nii_header['pixdim'][1:4]
    if verbose: print(f'The voxel size is {voxels_size[0]} x {voxels_size[1]} x {voxels_size[2]} mm^3')
    ### COMPUTING THE MINIMUM AND MAXIMUM SIZES OF THE IMAGE ###
    L_min = voxels_size[0]
    if verbose: print(f'Shape of the image : {imageloaded.shape}')
    if verbose: print(f'The minimum size of the image is {L_min} mm')
    Ly=imageloaded.shape[0]
    Lx=imageloaded.shape[1]
    Lz=imageloaded.shape[2]
    if Lx > Ly:
        L_Max = Lx
    else:
        L_Max = Ly
    if Lz > L_Max:
        L_Max = Lz
    if verbose: print(f'The maximum size of the image is {L_Max} mm')
    ### NON-ZERO VOXELS OF THE IMAGE: NUMBER AND Y, X, Z COORDINATES ###
    voxels=[]
    for i in range(Ly):
        for j in range(Lx):
            for k in range(Lz):
                if imageloaded[i,j,k]>0:
                    voxels.append((i,j,k))
    voxels=np.asarray(voxels)
    if verbose: print(f'The non-zero voxels in the image are (the image volume) {voxels.shape[0]} / {math.prod(imageloaded.shape)}')
    ### LOGARITHM SCALES VECTOR AND COUNTS VECTOR CREATION ###
    Ns = []
    scales = []
    stop = math.ceil(math.log2(L_Max))
    for exp in range(stop+1):
        scales.append(2**exp)
    scales = np.asarray(scales)
    random.seed(1)
    ### THE 3D BOX-COUNTING ALGORITHM WITH 20 PSEUDO-RANDOM OFFSETS ###
    for scale in scales:
        if verbose: print(f'Computing scale {scale}...')
        Ns_offset=[] 
        for i in range(20): 
            y0_rand = -random.randint(0,scale)
            yend_rand = Ly+1+scale
            x0_rand = -random.randint(0,scale)
            xend_rand = Lx+1+scale
            z0_rand = -random.randint(0,scale)
            zend_rand = Lz+1+scale
            # computing the 3D histogram
            H, edges=np.histogramdd(voxels, bins=(np.arange(y0_rand,yend_rand,scale), np.arange(x0_rand,xend_rand,scale), np.arange(z0_rand,zend_rand,scale)))
            Ns_offset.append(np.sum(H>0))
            if verbose: print(f'======= Offset {i+1}: x0_rand = {x0_rand}, y0_rand = {y0_rand}, z0_rand = {z0_rand}, count = {np.sum(H>0)}')
        Ns.append(np.mean(Ns_offset))
    ### AUTOMATED SELECTION OF THE FRACTAL SCALING WINDOW ### 
    minWindowSize = 5 # in the logarithm scale, in the worst case, 5 points cover more than 1.2 decades, which should be the minimum fractal scaling window possible, to define an object as fractal (Marzi et al., Scientific Reports 2020)
    scales_indices = [] 

    for step in range(scales.size, minWindowSize-1, -1):
        for start_index in range(0, scales.size-step+1):
            scales_indices.append((start_index, start_index+step-1))
    scales_indices = np.asarray(scales_indices)    

    k_ind = 1 # number of indipendent variables in the regression model
    R2_adj = -1
    for k in range(scales_indices.shape[0]):
        coeffs=np.polyfit(np.log2(scales)[scales_indices[k,0]:scales_indices[k,1] + 1], np.log2(Ns)[scales_indices[k,0]:scales_indices[k,1] + 1], 1)
        n = scales_indices[k,1] - scales_indices[k,0] + 1 
        y_true = np.log2(Ns)[scales_indices[k,0]:scales_indices[k,1] + 1]
        y_pred = np.polyval(coeffs,np.log2(scales)[scales_indices[k,0]:scales_indices[k,1] + 1])
        R2=skl.r2_score(y_true,y_pred)
        R2_adj_tmp = 1 - (1 - R2)*((n - 1)/(n - (k_ind + 1)))
        if verbose: print(f'In the interval [{scales[scales_indices[k,0]]}, {scales[scales_indices[k,1]]}] voxels, the FD is {-coeffs[0]} and the determination coefficient adjusted for the number of points is {R2_adj_tmp}')
        R2_adj = round(R2_adj, 3)
        R2_adj_tmp = round(R2_adj_tmp, 3)
        if R2_adj_tmp > R2_adj:
            R2_adj = R2_adj_tmp
            FD = -coeffs[0]
            mfs = scales[scales_indices[k,0]]
            Mfs = scales[scales_indices[k,1]]
            fsw_index = k
            coeffs_selected = coeffs
        FD = round(FD, 4)
    ### FRACTAL ANALYSIS RESULTS ###
    mfs = mfs * L_min
    Mfs = Mfs * L_min
    if verbose: print("mfs automatically selected:", mfs)
    if verbose: print("Mfs automatically selected:", Mfs)
    if verbose: print("FD automatically selected:", FD)

    return FD