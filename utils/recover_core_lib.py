import cv2
import numpy as np
import cupy as cp
from tqdm import tqdm
from scipy import signal
from pathlib import Path
from scipy.io.wavfile import write



# --------------- Visualization
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, TextBox

# launch_vibration_viewer(show_shifts=all_shifts_svd, SHOW_VIS="sig_zoom")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, TextBox

class VibrationViewer:
    def __init__(self, show_shifts, mode="xy_phase", n_points=6, sensors=36):
        self.show_shifts = show_shifts
        self.mode = mode
        self.n_points = n_points
        self.sensors = sensors
        self.time_steps = show_shifts.shape[1]
        self.init_time = 0

        self.slider = None
        self.q = None  # quiver object
        self.vline = None  # vertical line in plots

    def show(self):
        if self.mode == "xy_phase":
            self._show_xy_phase()
        elif self.mode == "sig_zoom":
            self._show_sig_zoom()

    def _show_xy_phase(self):
        x_coords, y_coords = np.meshgrid(np.arange(self.n_points), np.arange(self.n_points))
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(4, 2, height_ratios=[4, 1, 1, 0.3])

        ax_quiver = fig.add_subplot(gs[0, 0])
        ax_signal = fig.add_subplot(gs[0, 1])
        ax_slider = fig.add_subplot(gs[1, :])
        ax_scale_slider = fig.add_subplot(gs[2, :])

        vectors = self.show_shifts[:, self.init_time, :].reshape(self.n_points, self.n_points, 2)
        U, V = vectors[:, :, 0], vectors[:, :, 1]
        self.q = ax_quiver.quiver(x_coords, y_coords, U, V, pivot='tail', scale=1)
        ax_quiver.set_title(f"Surface Vibrations at Time Index {self.init_time}")
        ax_quiver.set_xlim(-0.5, self.n_points - 0.5)
        ax_quiver.set_ylim(-0.5, self.n_points - 0.5)
        ax_quiver.set_aspect('equal')
        ax_quiver.set_xlabel("Sensor X position")
        ax_quiver.set_ylabel("Sensor Y position")

        signal = self.show_shifts[0, :, 0]
        time_array = np.arange(self.time_steps)
        ax_signal.plot(time_array, signal, lw=1)
        ax_signal.set_title("Sensor 0 X-axis Signal")
        ax_signal.set_xlabel("Time Index")
        ax_signal.set_ylabel("Amplitude")
        self.vline = ax_signal.axvline(self.init_time, color='red', lw=1.5)

        self.slider = Slider(ax=ax_slider, label='Time Index',
                             valmin=0, valmax=self.time_steps - 1, valinit=self.init_time,
                             valfmt='%0.0f', valstep=1)
        self.slider.on_changed(lambda val: self._update_quiver(val, ax_quiver))

        scale_slider = Slider(ax=ax_scale_slider, label='Quiver Scale',
                              valmin=0.1, valmax=10, valinit=1, valstep=0.1)
        scale_slider.on_changed(lambda val: self._update_quiver(self.slider.val, ax_quiver, scale_slider.val))

        # Buttons
        ax_button_left = fig.add_subplot(gs[3, 0])
        ax_button_right = fig.add_subplot(gs[3, 1])
        btn_left = Button(ax_button_left, 'Step Left')
        btn_right = Button(ax_button_right, 'Step Right')

        btn_left.on_clicked(lambda event: self._step(-1))
        btn_right.on_clicked(lambda event: self._step(+1))

        # Text box for manual index
        ax_textbox = plt.axes([0.4, 0.02, 0.2, 0.04])
        text_box = TextBox(ax_textbox, "Jump to Index", initial=str(self.init_time))
        text_box.on_submit(self._manual_jump)

        plt.subplots_adjust(bottom=0.1)
        plt.show()

    def _update_quiver(self, t_index, ax_quiver, scale=1):
        t_index = int(t_index)
        vectors = self.show_shifts[:, t_index, :].reshape(self.n_points, self.n_points, 2)
        U, V = vectors[:, :, 0], vectors[:, :, 1]
        if self.q:
            self.q.remove()
        self.q = ax_quiver.quiver(np.arange(self.n_points).reshape(-1, 1),
                                  np.arange(self.n_points).reshape(1, -1),
                                  U, V, pivot='tail', scale=scale)
        self.vline.set_xdata([t_index])
        ax_quiver.set_title(f"Surface Vibrations at Time Index {t_index}")
        ax_quiver.figure.canvas.draw_idle()

    def _step(self, direction):
        current = int(self.slider.val)
        new = np.clip(current + direction, 0, self.time_steps - 1)
        self.slider.set_val(new)

    def _manual_jump(self, text):
        try:
            idx = int(text)
            if 0 <= idx < self.time_steps:
                self.slider.set_val(idx)
        except ValueError:
            pass

    def _show_sig_zoom(self):
        window_size = 1000
        half_window = window_size // 2

        def get_window_indices(t_index):
            return max(0, t_index - half_window), min(self.time_steps, t_index + half_window)

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1.5, 3, 3, 1])

        ax_full = fig.add_subplot(gs[0])
        time_array = np.arange(self.time_steps)
        ax_full.plot(time_array, self.show_shifts[0, :, 0], lw=1)
        self.vline = ax_full.axvline(0, color='red', lw=1.5)
        ax_full.set_title("Full Signal of Sensor 0 (X-axis)")
        ax_full.set_xlabel("Time Index")
        ax_full.set_ylabel("Amplitude")

        ax_sig_x = fig.add_subplot(gs[1])
        ax_sig_y = fig.add_subplot(gs[2])

        def update(val):
            t_index = int(self.slider.val)
            self.vline.set_xdata([t_index])
            start, end = get_window_indices(t_index)
            time_window = np.arange(start, end)

            ax_sig_x.cla()
            ax_sig_y.cla()
            for sensor in range(self.sensors):
                ax_sig_x.plot(time_window, self.show_shifts[sensor, start:end, 0], lw=1)
                ax_sig_y.plot(time_window, self.show_shifts[sensor, start:end, 1], lw=1)
            ax_sig_x.axvline(t_index, color='red', lw=1.5)
            ax_sig_y.axvline(t_index, color='red', lw=1.5)
            ax_sig_x.set_title("X-axis Signals (Zoomed)")
            ax_sig_x.set_ylabel("Amplitude")
            ax_sig_y.set_title("Y-axis Signals (Zoomed)")
            ax_sig_y.set_xlabel("Time Index")
            ax_sig_y.set_ylabel("Amplitude")
            fig.canvas.draw_idle()

        self.slider = Slider(ax=fig.add_subplot(gs[3]), label='Time Index',
                             valmin=0, valmax=self.time_steps - 1, valinit=self.init_time,
                             valfmt='%0.0f', valstep=1)
        self.slider.on_changed(update)

        ax_textbox = plt.axes([0.4, 0.02, 0.2, 0.04])
        text_box = TextBox(ax_textbox, "Jump to Index", initial=str(self.init_time))
        text_box.on_submit(self._manual_jump)

        plt.subplots_adjust(hspace=0.5, bottom=0.1)
        update(self.init_time)
        plt.show()

# --------------- MISC 


def save_signal_as_wav(filename, signal, sampling_rate):
    if np.issubdtype(signal.dtype, np.floating):
        signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    
    # Write to WAV file
    write(filename, sampling_rate, signal)
    print(f"Audio file saved as: {filename}")   

def replace_second_last_dir(path, new_dir):
    p = Path(path.rstrip("/"))  # Remove trailing slash if present
    new_path = p.parent.parent / new_dir / p.name  # Replace second-to-last dir
    return str(new_path)  # Convert to string without adding a trailing slash

def read_roi_matrix(file_path):
    """Reads a boolean matrix from a .txt file and converts it into a NumPy array."""
    with open(file_path, "r") as f:
        # Read file and evaluate as a Python list
        matrix = eval(f.read())  # Converts string representation into a list
    return np.array(matrix, dtype=bool)

def stretch_contrast(image,minmax=None):
    if minmax is None:
        min_val = np.min(image)
        max_val = np.max(image)
    else:
        min_val,max_val = minmax
    stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
    return stretched

def get_pad_size(w):
    two_power = 2**np.ceil(np.log2(w))
    left_pad  = int((two_power-w)//2)
    right_pad = int(two_power-w-left_pad)
    return left_pad,right_pad
def filter_signal(sig, run_opt, cutoff, numtaps=None, padding_size=None):
    
    if padding_size == None:
        padding_size = int(sig.shape[0] / 20) # the number 20 is a heuristic
    
    if numtaps == None:
        numtaps = int(sig.shape[0]) + 2*padding_size
        print(numtaps)
    
    numtaps = numtaps - 1 if numtaps % 2 == 0 else numtaps
    
    bandpass = signal.firwin(numtaps     = numtaps, # number of frames minus 1
                             cutoff      = cutoff,
                             pass_zero   = False, 
                             fs          = run_opt['cam_params']['camera_FPS'])

    transfer_function = np.fft.fft(bandpass)
    


    # Pad signal with zeros before FFT
    padded_signal = np.pad(sig, (padding_size, padding_size), mode='reflect')

    # Zero-pad the bandpass filter to match the length of the padded signal
    padded_signal_length = padded_signal.shape[0]  # Get the length of the padded signal

    # Zero-pad the bandpass filter to the length of the padded signal
    padded_bandpass = np.pad(abs(transfer_function), 
                             (0, padded_signal_length - len(abs(transfer_function))), 
                             mode='constant')

    # Apply FFT-based filtering using the padded bandpass filter
    filtered_signal = np.fft.ifft(padded_bandpass * np.fft.fft(padded_signal)).real

    # Remove padding after filtering
    return filtered_signal[padding_size:-padding_size]    

# --------------- Define ROIs for recovery

def select_points(event, x, y, flags, param):
    global points, num_points
    
    # On left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point
        points.append((x, y))
        
        # Draw a horizontal line across the row of the clicked point
        cv2.line(image, (x, 0), (x, image.shape[0]), 255, 2)
        cv2.imshow("Image", image)
        num_points += 1
        
        # If we have reached the desired number of points, stop
        if num_points >= param:
            cv2.setMouseCallback("Image", lambda *args : None)  # Disable further callbacks
            cv2.destroyAllWindows()
def get_points_from_image(image_np, N):
    global image, points, num_points
    points = []  # Reset points list
    num_points = 0  # Reset point counter
    image = image_np.copy()  # Work on a copy of the image
    
    # Display the image and set the mouse callback function
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", select_points, param=N)
    
    # This ensures the window can update properly
    while True:
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        # If all points are selected or the user presses the 'q' key, exit loop
        if num_points >= N or key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    return points


# --------------- RUN 
# CPU version functions:

# CPU VERSION
def create_hann_window(image_shape,margin,dtype =cv2.CV_32F):
    def pad_image(image, h_pad, v_pad):
        padded_image = cv2.copyMakeBorder(image, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image
    
    h,w  = image_shape
    hannW = cv2.createHanningWindow((w - 2*margin, h - 2*margin), dtype)
    hannW = pad_image(hannW,margin,margin)
    return hannW
def Parabola(mat):
    hei,wid = mat.shape
    boxsize = 3
    cy = int((hei-1)/2)
    cx = int((wid-1)/2)
    bs = int((boxsize-1)/2)
    box = mat[cy-bs:cy-bs+boxsize,cx-bs:cx-bs+boxsize]
    # [x^2 y ^2 x y 1]
    Tile = np.arange(boxsize,dtype=float)-bs
    Tx = np.tile(Tile,[boxsize,1])
    Ty = Tx.T
    Ones = np.ones((boxsize*boxsize,1),dtype=float)
    x = Tx.reshape(boxsize*boxsize,1)
    y = Ty.reshape(boxsize*boxsize,1)
    x2 = x*x
    y2 = y*y
    A = np.concatenate((x2,y2,x,y,Ones),1)
    # data = A^+ B
    data = np.dot(np.linalg.pinv(A) , box.reshape(boxsize*boxsize,1))
    # xmax = -c/2a, ymax = -d/2b, peak = e - c^2/4a - d^2/4b
    a,b,c,d,e = data.squeeze()
    Ay = -d /2.0/b
    Ax = -c /2.0/a
    #self.peak = e - c*c/4.0/a - d*d/4.0/b
    return [Ay,Ax]

def _phase_correlation_generic(a, b, hannw, do_interpolation=True):
    height, width = a.shape

    # Compute FFTs
    G_a = np.fft.fft2(a * hannw)
    G_b = np.fft.fft2(b * hannw)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
        
    # Inverse FFT and shift zero-frequency component to the center
    r = np.fft.fftshift(np.fft.ifft2(R).real)
    
    # Find the peak in the correlation
    DY, DX = np.unravel_index(r.argmax(), r.shape)
    
    if do_interpolation:
        # Subpixel refinement using a 3x3 box
        boxsize  = 3
        half_box = int((boxsize - 1) // 2)
        box = r[DY - half_box:DY + half_box + 1, DX - half_box:DX + half_box + 1]
        TY, TX = Parabola(box)
        sDY = TY + DY
        sDX = TX + DX
        shifts = np.array([math.floor(width/2) - sDX, math.floor(height/2) - sDY])
    else:
        shifts = np.array([math.floor(width/2) - DX, math.floor(height/2) - DY])
    
    # Return the negative shift values
    return -shifts[0], -shifts[1]

def phase_correlation(a, b, hannw):
    return _phase_correlation_generic(a, b, hannw, do_interpolation=True)
def phase_correlation_int(a, b, hannw):
    return _phase_correlation_generic(a, b, hannw, do_interpolation=False)

def warp(image, dx, dy):
    # Compute the 2D FFT of the image
    image_fft = np.fft.fft2(image)
    
    # Create the phase shift matrix
    rows, cols = image.shape
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    phase_shift = np.exp(-2j * np.pi * (V * dx + U * dy))
    
    # Apply the phase shift in the frequency domain
    shifted_image_fft = image_fft * phase_shift
    
    # Compute the inverse FFT to get the shifted image
    shifted_image = np.fft.ifft2(shifted_image_fft).real
    
    return shifted_image
def warp_int(image, dx, dy):
    # For integer shifts, np.roll is much faster.
    return np.roll(image, shift=(int(dy), int(dx)), axis=(0, 1))

def _find_frame_translation_PC_generic(frame_1, frame_2, phase_corr_fn):
    frame_1 = frame_1.astype('float64')/255
    frame_2 = frame_2.astype('float64')/255

    margin = 0
    hannW = create_hann_window(frame_1.shape, margin, dtype=cv2.CV_64F)

    def get_pad_size(w):
        two_power = 2**np.ceil(np.log2(w))
        left_pad  = int((two_power-w)//2)
        right_pad = int(two_power-w-left_pad)
        return left_pad, right_pad

    h, w = frame_1.shape
    left_pad, right_pad = get_pad_size(w)
    up_pad, down_pad   = get_pad_size(h)

    def pad(frame):
        return np.pad(frame, ((up_pad, down_pad), (left_pad, right_pad)))

    frame_1 = pad(frame_1)
    frame_2 = pad(frame_2)
    hannW   = pad(hannW)

    shift = phase_corr_fn(frame_2, frame_1, hannW)
    tx, ty = shift[0], shift[1]
    return tx, ty
def find_frame_translation_PC(frame_1, frame_2):
    return _find_frame_translation_PC_generic(frame_1, frame_2, phase_correlation)
def find_frame_translation_PC_int(frame_1, frame_2):
    return _find_frame_translation_PC_generic(frame_1, frame_2, phase_correlation_int)
def find_frame_translation_LKi(image1, image2, iterations=3):

    # Initialize the shift (translation vector)
    shift = np.array([0, 0], dtype=np.float64)

    aligned_image = image2.copy()
    
    for _ in range(iterations):
        # Compute the image gradients
        I_x = 0.5 * (image1[1:-1, 2:].astype(np.float64) - image1[1:-1, :-2].astype(np.float64))
        I_y = 0.5 * (image1[2:, 1:-1].astype(np.float64) - image1[:-2, 1:-1].astype(np.float64))
        I_t = (aligned_image.astype(np.float64) - image1.astype(np.float64))[1:-1, 1:-1]

        # Flatten the gradients to form the system of equations
        I_x = I_x.flatten()
        I_y = I_y.flatten()
        I_t = I_t.flatten()

        # Construct the matrix A and vector b for the least squares problem
        A = np.vstack((I_x, I_y)).T
        b = -I_t
        
        # Solve the least squares problem to find the update in (dx, dy)
        delta_shift, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Warp the second image according to the current shift estimate
        aligned_image = warp(aligned_image, -1*delta_shift[0], -1*delta_shift[1])

        # Update the shift vector (dx, dy)
        shift += delta_shift     

    # The final estimated translation
    tx, ty = shift
    return tx, ty
def find_frame_translation_PCLKi(image1, image2, iterations=3, IS_FAST=False):
    
    if IS_FAST:
        phase_corr_tx, phase_corr_ty = find_frame_translation_PC_int(image1, image2)
        image2_aligned = warp_int(image2, -phase_corr_tx, -phase_corr_ty)
    else:
        phase_corr_tx, phase_corr_ty = find_frame_translation_PC(image1, image2)
        image2_aligned = warp(image2, -phase_corr_tx, -phase_corr_ty)
                
    LK_tx, LK_ty     = find_frame_translation_LKi(image1, image2_aligned, iterations)
    
    tx,ty            = phase_corr_tx + LK_tx, phase_corr_ty + LK_ty
    M                = np.hstack((np.eye(2),np.array([[tx],[ty]])))
    shift = (tx,ty)
    
    ret = 1 #<--FIX THAT
    return shift, M, ret
def compute_CAM2_translations_v1_orig(video_pp_cam2,debug=0, IS_FAST=False):
    
    n_ref_frames               = video_pp_cam2.shape[0]
    all_reference_shifts       = []
    succ_frames                = np.zeros((n_ref_frames,),dtype=bool)

    print('----- Computing shifts -----')
    
    if debug:
        video_pp_cam2_debug    = []
        video_pp_cam2_debug.append(video_pp_cam2[0])
    else:
        video_pp_cam2_debug    = None

    M_total                    = np.eye(2, 3, dtype=np.float32)
    succ_frames[0]             = True
    all_reference_shifts.append([0,0])  
    
    prev = 0
    for i in tqdm(range(1,n_ref_frames)):
        reference_shift, M, ret = find_frame_translation_PCLKi(video_pp_cam2[prev].copy(),video_pp_cam2[i].copy(),IS_FAST=IS_FAST)

        if ret==1:
            succ_frames[i]      = True
            all_reference_shifts.append(reference_shift)
            prev                = i 
            M_total[:,-1] += M[:,-1]
            if debug:
                video_pp_cam2_debug.append( cv2.warpAffine(video_pp_cam2[i].copy(), M_total ,video_pp_cam2[0].shape[::-1]))
        else:
            if debug:
                video_pp_cam2_debug.append( video_pp_cam2[i])
            print(f"frame {i} failed")
            
    all_shifts = np.array(all_reference_shifts).squeeze().cumsum(axis=0)

    return all_shifts, np.array(video_pp_cam2_debug),succ_frames


# CUPY VERSION
def compute_CAM2_translations_v3_cupy(video_pp_cam2, batch_size = 128):
    
    n_ref_frames = video_pp_cam2.shape[0]
    # Preallocate a cupy array for all_reference_shifts: one shift per frame (first is [0,0])
    all_reference_shifts    = cp.empty((n_ref_frames, 2), dtype=cp.float32)
    all_reference_shifts[0] = cp.array([0, 0], dtype=cp.float32)

    print('----- Computing shifts -----')
    
    # There are (n_ref_frames - 1) shifts, process them in batches
    N_batches = int(np.ceil((n_ref_frames - 1) / batch_size))
    
    for i in tqdm(range(N_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_ref_frames - 1)
        # For each batch, we need frames from index start to end+1
        batch_frames = video_pp_cam2[start : end + 2]
        # Compute shifts for this batch; expected shape: (number_of_shifts_in_batch, 2)
        reference_shifts_batch = find_frame_translation_PCLKi_cupy(batch_frames)
        # Place the computed shifts into our preallocated array
        all_reference_shifts[start + 1 : end + 2] = reference_shifts_batch

    # Compute the cumulative sum over shifts and negate to get the alignment shifts per frame
    all_shifts = cp.cumsum(all_reference_shifts, axis=0)
    # Convert the final result to a numpy array
    all_shifts = cp.asnumpy(all_shifts)
    
    return all_shifts, None
def find_frame_translation_PCLKi_cupy(video, iterations=3):
    
    shifts_PC,video_cp     = find_frame_translation_PC_cupy(video)
    # shifts = (127, 2), video_cp =  (128, 6, 120)
    video_aligned_roll     = warp_roll(video_cp, -shifts_PC)
        
    shifts_LK              = find_frame_translation_LKi_cupy(video_aligned_roll, iterations)
    
    return shifts_PC+shifts_LK
def find_frame_translation_PC_cupy(video):
    
    video_cp      = cp.asarray(video, dtype=cp.float32) / 255
    
    N,h,w         = video.shape
    hannW         = cp.outer(cp.hanning(h), cp.hanning(w))

    left_pad,right_pad = get_pad_size(w)
    up_pad,  down_pad  = get_pad_size(h)
        
    def pad(arr):
        if arr.ndim == 2:
            return cp.pad(arr, ((up_pad, down_pad), (left_pad, right_pad)))
        elif arr.ndim == 3:
            return cp.pad(arr, ((0, 0), (up_pad, down_pad), (left_pad, right_pad)))
        else:
            raise ValueError("Unsupported array dimensions for padding")     
    
    hannW         = pad(hannW)
    
    shifts        = phase_correlation_cupy(pad(video_cp),hannW)

    return shifts, video_cp
def phase_correlation_cupy(video, hannw):
    # video: (N, h, w), hannw: (h, w)
    eps = 1e-8
    
    N, h, w = video.shape

    # Apply the window and compute FFT for every frame at once.
    video_fft = cp.fft.fft2(video * hannw, axes=(-2, -1))

    # Compute the normalized cross-power spectrum between consecutive frames.
    R = video_fft[:-1] * cp.conj(video_fft[1:])
    R /= (cp.abs(R) + eps)

    # Inverse FFT to get the cross-correlation (batch mode)
    corr = cp.fft.ifft2(R, axes=(-2, -1)).real
    corr = cp.fft.fftshift(corr, axes=(-2, -1))

    # Flatten each (h, w) correlation map and find the index of its maximum.
    corr_flat = corr.reshape(corr.shape[0], -1)
    max_idx = cp.argmax(corr_flat, axis=1)
    peak_row = max_idx // w  # y-coordinate
    peak_col = max_idx % w   # x-coordinate

    # Compute shifts: if the peak is at the center, shift=0.
    shift_x = peak_col - (w // 2)
    shift_y = peak_row - (h // 2)
    shifts = cp.stack([shift_x, shift_y], axis=1)
    
    return -shifts
def warp_roll(video, shifts):
    aligned_video = video.copy()
    cumulative_shift = cp.array([0, 0], dtype=cp.int32)
    
    for i in range(1, video.shape[0]):
        cumulative_shift += shifts[i - 1]
        # Convert cumulative shift components to Python ints for cp.roll.
        dx = int(cumulative_shift[0].item())
        dy = int(cumulative_shift[1].item())
        # Note: cp.roll expects shifts as (shift_along_axis0, shift_along_axis1) which corresponds to (dy, dx)
        aligned_video[i] = cp.roll(video[i], shift=(dy, dx), axis=(0, 1))
    
    return aligned_video
def warp_video_fft(video, shifts):
    
    N, h, w = video.shape

    # Compute FFT for each frame (batch FFT over the last two axes)
    video_fft = cp.fft.fft2(video, axes=(-2, -1))

    # Create frequency grids for the spatial dimensions
    u = cp.fft.fftfreq(h)[:, None]  # shape: (h, 1)
    v = cp.fft.fftfreq(w)[None, :]   # shape: (1, w)

    # Compute phase shifts for each frame using broadcasting:
    # shifts[:, 0] is dx (applied to the horizontal frequencies v)
    # shifts[:, 1] is dy (applied to the vertical frequencies u)
    phase_shifts = cp.exp(-2j * cp.pi * (shifts[:, 0][:, None, None] * v +
                                         shifts[:, 1][:, None, None] * u))
    
    # Apply the phase shift to each frame in the Fourier domain
    shifted_video_fft = video_fft * phase_shifts

    # Compute the inverse FFT to obtain the shifted video
    shifted_video = cp.fft.ifft2(shifted_video_fft, axes=(-2, -1)).real

    return shifted_video
def find_frame_translation_LKi_cupy(video, iterations=3):

    # Define frame pairs: image1 and image2 (for N-1 pairs)
    image1 = video[:-1]  # shape (B, h, w)
    image2 = video[1:]   # shape (B, h, w)
    B, h, w = image1.shape
    
    # Initialize aligned_image (the moving image) and shift estimates.
    aligned_image = image2.copy()  # shape (B, h, w)
    shift = cp.zeros((B, 2), dtype=cp.float32)
    
    # Precompute spatial gradients of image1.
    # We restrict to the inner region [1:-1,1:-1] to avoid boundary issues.
    I_x = 0.5 * (image1[:, 1:-1, 2:] - image1[:, 1:-1, :-2])    # shape (B, h-2, w-2)
    I_y = 0.5 * (image1[:, 2:, 1:-1] - image1[:, :-2, 1:-1])    # shape (B, h-2, w-2)
    
    for _ in range(iterations):
        # Compute temporal error over the same inner region.
        I_t = aligned_image[:, 1:-1, 1:-1] - image1[:, 1:-1, 1:-1]  # shape (B, h-2, w-2)
        
        # Compute the sums required for the 2x2 normal equation per frame.
        sum_Ix2  = cp.sum(I_x * I_x, axis=(1, 2))
        sum_Iy2  = cp.sum(I_y * I_y, axis=(1, 2))
        sum_IxIy = cp.sum(I_x * I_y, axis=(1, 2))
        sum_IxIt = cp.sum(I_x * I_t, axis=(1, 2))
        sum_IyIt = cp.sum(I_y * I_t, axis=(1, 2))
        
        # Solve for the incremental shift using the closed-form solution:
        # [sum_Ix2   sum_IxIy] [delta_x] = -[sum_IxIt]
        # [sum_IxIy  sum_Iy2 ] [delta_y]   -[sum_IyIt]
        det = sum_Ix2 * sum_Iy2 - sum_IxIy ** 2 + 1e-8  # Avoid division by zero
        
        delta_x = (- sum_IxIt * sum_Iy2 + sum_IxIy * sum_IyIt) / det
        delta_y = (- sum_IyIt * sum_Ix2 + sum_IxIy * sum_IxIt) / det
        
        delta = cp.stack([delta_x, delta_y], axis=1)  # shape (B, 2)
        
        # Accumulate the incremental shift estimates.
        shift += delta
        
        # Warp the aligned_image using the negative of the computed delta shifts.
        # This warps each frame in the batch simultaneously.
        aligned_image = warp_video_fft(aligned_image, -delta)
    
    return shift