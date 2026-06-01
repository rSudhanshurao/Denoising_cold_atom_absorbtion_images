#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import sif_parser
from scipy.ndimage import gaussian_filter
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import time
from scipy.signal import correlate2d
import os

class ImageDefringing:
    def __init__(self,mode='auto'):
        self.mode=mode
        self.light_params = {'threshold_rel': 0.15, 'num_peaks': 5}
        self.strong_params = {'threshold_rel': 0.05, 'num_peaks': 15}
    #Private methods
    #Preprocessing

    def _Crop_Image(self,img,box_size=85):  #Image is cropped to 170*170 pixels
        """
        Definition-Gives the bounds for cropping
        Input- Image, size of the cropped box.
        Output- Bounds of the cropped box

        """
        if img.sum()<=0:
            return 85, 0, 85, 0
        y_center,x_center=center_of_mass(img) #Center_of_mass gives center of the box
        cy, cx = int(round(y_center)), int(round(x_center))#changing the coordinates of center to an integr
        ymax, ymin=cy+box_size,cy-box_size #y bounds
        xmax,xmin=cx+box_size,cx-box_size#x bounds
        return ymax,ymin,xmax,xmin

    def _Preprocess(self,data):
        """
        Definition-Cleans the image, crops it and normalizes it.
        Input-Data file containing the images
        Output-Signal and Reference images
        """
        #Defining signal
        try:
            d, md = sif_parser.np_open(data) #d contains images and md contains the background information 
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: '{data}'") #Error if file is not present
        except Exception as e:
            raise RuntimeError(f"Failed to read .sif file '{data}': {e}")
        # Checking if file has signal, sufficient references and backgrounds
        if d.shape[0] < 3:
            raise ValueError(f"Expected at least 3 frames, got {d.shape[0]}")
        if d.shape[0] < 37:
            print(f"WARNING: Expected 37 frames, got {d.shape[0]} — "
                f"fewer references may be available")
        s=d[0] #Extracting signal from the data
        s_clean=s-d[2]   #Subtracts background from signal
        if s_clean.max()<0:   
            raise ValueError("Signal after background subtraction giving negative values, check background frame")
        r_clean=[]
        for i in range(1,36):   #Extracting the 18 reference images
            if i%2!=0:
                bg=d[i]-d[i+1]       #Subtracts background from reference
                r_clean.append(bg)

    #Cropping the image
        s_y_max,s_y_min,s_x_max,s_x_min=self._Crop_Image(s_clean,box_size=85)   #cropping the signal image to the given size
        #Checking for edge cases
        H, W = s_clean.shape
        s_y_min = max(0, s_y_min)
        s_y_max = min(H, s_y_max)
        s_x_min = max(0, s_x_min)   
        s_x_max = min(W, s_x_max)
        s_cropped=s_clean[s_y_min:s_y_max,s_x_min:s_x_max]    
        
        r_cropped=[]               
        for img in r_clean:           #Cropping the reference image
            
            img_cropped=img[s_y_min:s_y_max,s_x_min:s_x_max]
            r_cropped.append(img_cropped)

        #Frobenius normalizing
        #Normalizing the signal image(frobenius)
        norm = np.linalg.norm(s_cropped, ord='fro')
        if norm == 0:
            raise ValueError("Cannot normalise — image has zero Frobenius norm")
        s_normalized=s_cropped/norm 
        r_normalized=[]
        for img in r_cropped:   #Normalizing the reference image
            norm = np.linalg.norm(img, ord='fro')
            if norm == 0:
                raise ValueError("Cannot normalise — image has zero Frobenius norm")
            img_norm=img/norm  
            r_normalized.append(img_norm)
        num_of_ref=len(r_normalized)  #number  of images
        
        return s_normalized,r_normalized,num_of_ref

    #Calculating Optical density
    def _calc_OD(self,signal, reference):
        """
        Definition-Calculates Optical density from signal and reference.
        Input-Signal and Reference image
        Output-Optical density
        """
        reference = np.clip(reference, 1e-10, None)
        ratio= np.clip(signal / reference, 1e-10, None)  #Background noise is already removed
        return -np.log(ratio)   #Returns the optical density

    #Defining the fitting function(gaussian)
    def _Gauss(self,x,A, x0, sx, B):
        """
        Definition-A Gaussian function for curvefitting
        Input-x coordinate, amplitude, initial position, standard deviation, error term
        Output-the gaussian function value
        """
        z=A * np.exp(-(((x - x0)**2) / (2 * sx**2)))+B   
        return z

    #Calculating average OD and average reference image
    def _necessary_calc(self,sig,ref,num_of_ref):
        """
        Definition-Calculates average OD and average reference image
        Input-Signal, reference and number of reference images
        Output-Optical density
        """
        h, w = sig.shape   #Size of the image
        OD_average  = np.zeros((h, w))
        Ref_average = np.zeros((h, w))
        OD=[] #Calculating average OD
        for img in ref:
            OD_i=self._calc_OD(sig,img)
            OD.append(OD_i)
        for i in OD:
                OD_average+=i
        OD_average/=num_of_ref
        #Calculating average reference image
        for img in ref:
            Ref_average+=img
        Ref_average/=num_of_ref
        return OD_average,Ref_average

    #Masking
    def _Masking(self,Sig, Ref,OD_average):
        """
        Definition-Masks the atoms to prevent it from being disturbed
        Input-Signal, reference,the average optical density
        Output-Optical density
        """
        h, w = Sig.shape
        #Extracting the 1D cross-section profiles
        x_profile = np.sum(OD_average, axis=0) 
        y_profile = np.sum(OD_average, axis=1)
        #Defining axes to fit curve
        x_axis=np.arange(len(x_profile))
        y_axis=np.arange(len(y_profile))

        #Creating guesses for the curve_fit function    
        guess_x = [np.max(x_profile), np.argmax(x_profile), 10, np.median(x_profile)]
        guess_y = [np.max(y_profile), np.argmax(y_profile), 10, np.median(y_profile)]

        #Curve fitting
        try:
            popt_x,_=curve_fit(self._Gauss,x_axis,x_profile,p0=guess_x)
            popt_y,_=curve_fit(self._Gauss,y_axis,y_profile,p0=guess_y)
            x0, sigma_x= popt_x[1],popt_x[2]
            y0, sigma_y= popt_y[1],popt_y[2]
        except RuntimeError:
            h, w   = OD_average.shape
            x0, y0 = w // 2, h // 2
            sigma_x  = min(h, w) // 6   #Incase of error
            sigma_y  = min(h, w) // 6   
            print("WARNING: Gaussian fit failed — using image centre as fallback")
        

        #Defining radius of mask
        sigma=max(abs(sigma_x),abs(sigma_y))
        mask_radius=2.5*sigma   #2.5*sigma is used here for optimal masking
        Mask_details=[x0,y0,mask_radius]

        #Replacing the masked area with the background of same area
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        distance_matrix = np.sqrt((X - x0)**2 + (Y - y0)**2)
        h, w = Sig.shape
        max_allowed_radius = min(h, w) // 3
        if mask_radius > max_allowed_radius:
            print(f"WARNING: mask_radius={mask_radius:.1f} clipped to {max_allowed_radius}")
            mask_radius = max_allowed_radius

        mask = distance_matrix <= mask_radius   #Creates a boolean matrix 
        if mask.sum() > 0.7 * h * w:
            raise ValueError("Mask covers more than 70% of image — "
                            "check Gaussian fit or cloud size")
        Masked_Signal=np.copy(Sig) #Creating a copy of the Signal to create the masked signal

        img=Ref[0]  #The first reference is used to mask
        noise1=img[mask]   #Storing the masked reference area
        Masked_Signal[mask]=noise1  #Replacing cloud with the reference mask
        return Masked_Signal,mask,Mask_details

    #Principle Component Analysis
    def _PCAalgo(self,sig,ref,ref_av,mask):
        """
        Definition-Extracts principal components and recreates the new reference image
        Input-Signal, reference, average reference, masked signal, mask truth values
        Output-Normalized principal components, new reference, PCA cleaned OD
        """
        A = []  #Defining the reference matrix(all references flattened and stacked together)
        Reference_average_flat = ref_av.flatten()
        for img in ref:
            A_i = np.copy(img).flatten()
            A_i -= Reference_average_flat  #Mean centering the references
            A.append(A_i)
        A = np.column_stack(A)  #Creating the column-stacked flattened reference matrix
        if np.allclose(A, A[:, 0:1]):
            raise ValueError("All reference frames are identical — "
                            "no fringe variation to decompose")
        #Defining the covariance matrix
        P_cov = np.matmul(np.transpose(A),A)
        P_cov=np.array(P_cov)
        #Extracting eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(P_cov)
        #Sorting eigenvectors based on eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[idx]
        eigenvectors_sorted = eigenvectors[:, idx]
        #Removing the insignificant eigenvectors
        threshold = (eigenvalues_sorted.max()) / 100
        cutoff = eigenvalues_sorted > threshold
        if cutoff.sum() == 0:
            print("WARNING: No PCs above threshold — lowering threshold")
            threshold  = eigenvalues_sorted.max() / 1000
            cutoff     = eigenvalues_sorted > threshold
        if cutoff.sum() == 0:
            raise ValueError("No valid principal components found")
        eigenvectors_rel = eigenvectors_sorted[:, cutoff]
        #Normalizing eigenvectors
        eigenvectors_rel=eigenvectors_rel/np.linalg.norm(eigenvectors_rel,axis=0)
        #Projecting eigenvectors on to the reference matrix
        PCs_raw = np.matmul(A, eigenvectors_rel) 
        #normalizing principal components
        PCs_normalized = PCs_raw / np.linalg.norm(PCs_raw, axis=0)
    
    #Creating the new reference image
        h, w = sig.shape
        num_pcs = PCs_normalized.shape[1]
        Signal_bg_sub = sig - Reference_average_flat.reshape(h, w)
    # Reshape PCs into a 3D block so it can be cut into blocks
        PCs_3D = PCs_normalized.reshape((h, w, num_pcs))
    # Create an empty canvas for the new, flexible reference
        X_new_tiled = np.zeros((h, w))
    # Define the Grid (2x2 grid = 85x85 patches)
        patch_size = 85
    # Loop through the image in patches
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
            # Calculate actual boundaries (prevents errors on edges)
                end_i = min(i + patch_size, h)
                end_j = min(j + patch_size, w)
            # Extract the local slices
                sig_patch = Signal_bg_sub[i:end_i, j:end_j].flatten()
                mask_patch = mask[i:end_i, j:end_j].flatten()
                pc_patch = PCs_3D[i:end_i, j:end_j, :].reshape(-1, num_pcs)
            # Find the background ONLY in the specific square
                bg_idx = ~mask_patch
            # Performing the Local Least Squares Fit
                if np.sum(bg_idx) > num_pcs:
                    local_weights, _, _, _ = np.linalg.lstsq(
                        pc_patch[bg_idx], 
                        sig_patch[bg_idx], 
                        rcond=None
                )
                else:
                # If a patch is 100% covered by atoms, weights are zero
                    local_weights = np.zeros(num_pcs)
            # Reconstructing the patch with the weights
                reconstructed_patch = pc_patch @ local_weights
            
                X_new_tiled[i:end_i, j:end_j] = reconstructed_patch.reshape(end_i - i, end_j - j)
        #Restore the baseline
        X_new = X_new_tiled + Reference_average_flat.reshape(h, w)
        #Calculating PCA clean OD
        PCA_OD_clean=self._calc_OD(sig,X_new)
        return PCs_normalized,X_new,PCA_OD_clean

    #Creating the fourier filter
    def _Fourierfilter(self,PCA_OD_clean):
        """
        Definition-Maps the fourier space of the OD, cleans the unwanted noises
        Input-PCA_cleaned OD
        Output-PCA+FFT cleaned OD,fourier space of the PCA cleaned OD,noise peaks
        """
        #Transforming the PCA cleaned OD into the fourier space and also shifting highest value to center
        f_OD = fftshift(fft2(PCA_OD_clean))
        magnitude_spectrum = np.log(np.abs(f_OD) + 1)
        rows, cols = magnitude_spectrum.shape
    #Protecting center from being erased
        center_y, center_x = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        #Defining the distance function
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        Masked_Space = np.copy(magnitude_spectrum)
        #Masking all pixels which are within a 25unit radius circle
        Masked_Space[dist_from_center <= 25] = 0
        #Detecting peaks
        peaks = peak_local_max(Masked_Space, min_distance=10, threshold_rel=0.1, num_peaks=8)
        if len(peaks) == 0:
            print("INFO: No fringe peaks detected in Fourier space — "
                "image may already be clean")
            return PCA_OD_clean, magnitude_spectrum, peaks
        fft_mask = np.ones((rows, cols))
    #Using FWHM
        for y, x in peaks:
            window_size = 25
            #Defining window endpoints
            y_min, y_max = max(0, y - window_size), min(rows, y + window_size + 1)
            x_min, x_max = max(0, x - window_size), min(cols, x + window_size + 1)
            #Creating a local window around the peaks
            local_window = Masked_Space[y_min:y_max, x_min:x_max]
            #Calculating threshold
            peak_intensity = Masked_Space[y, x]
            threshold = peak_intensity * 0.5
            y_grid, x_grid = np.ogrid[0:local_window.shape[0], 0:local_window.shape[1]]
            #Determing the center of the window
            center_y_local, center_x_local = local_window.shape[0] // 2, local_window.shape[1] // 2
            #Defining distance function
            dist_to_local_peak = np.sqrt((x_grid - center_x_local)**2 + (y_grid - center_y_local)**2)
            cage_radius = 12.0 # Maximum allowed spread
            #Applying the Caged Contour
            local_mask = fft_mask[y_min:y_max, x_min:x_max]
            # A pixel is only deleted if it is BOTH brighter than the threshold AND inside the cage
            noise_pixels = (local_window > threshold) & (dist_to_local_peak <= cage_radius)
            local_mask[noise_pixels] = 0
        #Smoothening the masked fourier space
        soft_mask = gaussian_filter(fft_mask, sigma=2)
        #Projecting the masked area to  the fourier space of OD
        F_OD_cleaned = f_OD * soft_mask
        #Transforming back to normal coordinates
        OD_fft_clean = np.real(ifft2(ifftshift(F_OD_cleaned)))
        return OD_fft_clean,magnitude_spectrum,peaks

    #Baseline correction
    def _Baseline_correction(self,PCA_FFT_clean_OD):
        """
        Definition-Corrects the baseline value of the final OD
        Input-Fourier space of the PCA cleaned OD, PCA+FFT cleaned OD
        Output-Baseline corrected PCA+FFT cleaned OD
        """
        rows, cols = PCA_FFT_clean_OD.shape
        corner_mask = np.zeros((rows, cols), dtype=bool)

        corner_mask[0:20,  0:20]   = True   # Bottom-Left
        corner_mask[0:20,  -20:]   = True   # Bottom-Right
        corner_mask[-20:,  0:20]   = True   # Top-Left
        corner_mask[-20:,  -20:]   = True   # Top-Right

        # Measure and subtract the global laser intensity fluctuation
        baseline_offset = np.mean(PCA_FFT_clean_OD[corner_mask])
        if abs(baseline_offset) > 0.5:
            print(f"WARNING: Large baseline offset {baseline_offset:.4f} — "
                f"corner region may overlap with cloud")
        OD_fft_clean_corrected = PCA_FFT_clean_OD - baseline_offset  #Subtracting the offset
        return OD_fft_clean_corrected
    

    def _Stat(self,file,OD_av,OD_clean):
        """
        Definition-Function used to give the statistics and results of the cleaning
        Input-Original sif file, Uncleaned optical density, final cleaned optical density 
        Output-Displays the plots and returns the standard deviation of the gaussian fit for both x and y axes
        """
        h,w=OD_clean.shape
        # ── Find peak of clean OD ──────────────────────────────────────────────────
        y_peak, x_peak = np.unravel_index(np.argmax(OD_clean), OD_clean.shape)  
        x_axis = np.arange(w)
        y_axis = np.arange(h)

        # ── Extract 1D slices ──────────────────────────────────────────────────────
        x_raw   = OD_av[y_peak, :]
        x_clean = OD_clean[y_peak, :]
        y_raw   = OD_av[:, x_peak]
        y_clean = OD_clean[:, x_peak]

        # ── Gaussian fits ──────────────────────────────────────────────────────────
        try:
            popt_x, pcov_x = curve_fit(self._Gauss, x_axis, x_clean,
                                        p0=[x_clean.max(), x_clean.argmax(), 20, 0])
            popt_y, pcov_y = curve_fit(self._Gauss, y_axis, y_clean,
                                        p0=[y_clean.max(), y_clean.argmax(), 20, 0])
        except RuntimeError:
            print("WARNING: Gaussian fit failed in _Stat — returning None")
            return None, None
        perr_x = np.sqrt(np.diag(pcov_x))
        perr_y = np.sqrt(np.diag(pcov_y))

        # ── Plot 1: X cross-section ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(x_axis, x_raw,   color='red',    alpha=0.4, linewidth=1,
                    label='Raw OD (with fringes)')
        axes[0].plot(x_axis, x_clean, color='blue',   linewidth=2,
                    label='Clean OD')
        axes[0].plot(x_axis, self._Gauss(x_axis, *popt_x), color='orange',
                    linewidth=2, linestyle='--',
                    label=f'Gaussian fit  σ={abs(popt_x[2]):.1f}±{perr_x[2]:.1f} px')
        axes[0].axhline(0, color='gray', linestyle=':', linewidth=1)
        axes[0].set_xlabel("Pixel position (x)", fontsize=13)
        axes[0].set_ylabel("Optical Density", fontsize=13)
        axes[0].set_title("Horizontal cross-section", fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # ── Plot 2: Y cross-section ────────────────────────────────────────────────
        axes[1].plot(y_axis, y_raw,   color='red',    alpha=0.4, linewidth=1,
                    label='Raw OD (with fringes)')
        axes[1].plot(y_axis, y_clean, color='blue',   linewidth=2,
                    label='Clean OD')
        axes[1].plot(y_axis, self._Gauss(y_axis, *popt_y), color='orange',
                    linewidth=2, linestyle='--',
                    label=f'Gaussian fit  σ={abs(popt_y[2]):.1f}±{perr_y[2]:.1f} px')
        axes[1].axhline(0, color='gray', linestyle=':', linewidth=1)
        axes[1].set_xlabel("Pixel position (y)", fontsize=13)
        axes[1].set_ylabel("Optical Density", fontsize=13)
        axes[1].set_title("Vertical cross-section", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("1D Cross-Section Benchmark: Raw vs Clean OD", fontsize=15)
        plt.tight_layout()
        base = os.path.splitext(file)[0]
        plt.savefig(f'{base}_stats.png')
        plt.close()
        return popt_x,popt_y

        #public functions
    def atom_denoising(self,file):
        """
        Definition-Defringes the images in the file
        Input-Unclean images file containing Signal, Reference and background
        Output-Clean optical density, dictionary containing results of defringing
        """
        #Main calling function for the file
        t0 = time.time()
        # Calling all functions
        Signal, Reference,nor=self._Preprocess(file)
        OD_average,Reference_average=self._necessary_calc(Signal,Reference,nor)
        Masked_Signal,mask,Mask_details=self._Masking(Signal,Reference,OD_average,170)
        PCs_normalized,X_new,OD_clean=self._PCAalgo(Signal,Reference,Reference_average,mask)
        OD_fft_clean,magnitude_spectrum,peaks=self._Fourierfilter(OD_clean)
        OD_fft_clean_corrected=self._Baseline_correction(OD_fft_clean)
        #Replacing invalid pixels with zero
        if np.any(np.isnan(OD_fft_clean_corrected)) or np.any(np.isinf(OD_fft_clean_corrected)):
            n_bad = np.sum(~np.isfinite(OD_fft_clean_corrected))
            print(f"WARNING: {n_bad} NaN/Inf pixels in output — replacing with 0")
            OD_fft_clean_corrected = np.nan_to_num(OD_fft_clean_corrected,
                                                    nan=0.0, posinf=0.0, neginf=0.0)
        #Plotting the final cleaned optical density
        plt.figure(figsize=(6,5))
        plt.imshow(OD_fft_clean_corrected, cmap='magma', origin='lower', vmin=0, vmax = OD_fft_clean_corrected.max()+0.1)
        plt.colorbar()
        plt.title("Enhanced Clean image")
        base = os.path.splitext(file)[0]
        plt.savefig(f'{base}_cleaned.png')
        plt.close()
        #Compile per-shot statistics for CSV logging
        popt_x, popt_y = self._Stat(file, OD_average, OD_fft_clean_corrected)
        if popt_x is None:
            print("WARNING: Statistics unavailable — Gaussian fit failed")
            return OD_fft_clean_corrected, None 
        current_stats = {"Shot_Name": 0,
                         "Time": 0,
        "Peak_OD_X": round(popt_x[0], 4),
        "Peak_OD_Y": round(popt_y[0], 4),
        "Sigma_X_px": np.abs(popt_x[2]),  
        "Sigma_Y_px": np.abs(popt_y[2]),  
        "Aspect_Ratio": round(abs(popt_y[2]) / abs(popt_x[2]), 3),
        "Center_X_px": round(popt_x[1], 2),
        "Center_Y_px": round(popt_y[1], 2),

        # Baseline Offsets
        "Baseline_X": round(popt_x[3], 4),
        "Baseline_Y": round(popt_y[3], 4),
        
        # Atom Number Proxy (Sum of OD inside the mask)
        "Total_OD_Sum": round(np.sum(OD_fft_clean_corrected[mask]), 2),
        
        # Algorithm Performance Metrics
        "Noise_Floor_Raw": round(np.std(OD_average[~mask]), 5),
        "Noise_Floor_Clean": round(np.std(OD_fft_clean_corrected[~mask]), 5),
        "Suppression_Factor": round(np.std(OD_average[~mask]) / np.std(OD_fft_clean_corrected[~mask]), 2),
        "PCs_Used": PCs_normalized.shape[1],
        "Fringe_Peaks_Found": len(peaks)
    }
        elapsed = (time.time() - t0) * 1000
        print(f"Processing time: {elapsed:.0f} ms")
        return OD_fft_clean_corrected,current_stats
