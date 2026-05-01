import cv2
import numpy as np
import math
from scipy.stats import binom
import os

# --- CONSTANTS FROM THE PAPER ---
M_PI = np.pi
M_LN10 = np.log(10)
NOTDEF = -1024.0  # Undefined angle (gradient too low)
NOTUSED = 0       # Pixel not used in region growing
USED = 1          # Pixel already used
LOG_EPS = 0       # Validation threshold: -log10(NFA) > 0 means NFA < 1

class LSD:
    def __init__(self, scale=0.8, quant=2.0, ang_th=22.5, density_th=0.7, n_bins=1024):
        """
        Implementation of "LSD: a Line Segment Detector" by Grompone von Gioi et al.
        """
        self.scale = scale
        self.sigma_scale = 0.6 
        self.quant = quant
        self.ang_th = ang_th
        self.density_th = density_th
        self.n_bins = n_bins
        
        # Pre-compute thresholds and probabilities
        self.prec = M_PI * self.ang_th / 180.0
        self.p_val = self.ang_th / 180.0 
        self.rho = self.quant / math.sin(self.prec) 

    def run(self, frame, fast_mode=False, enhance_contrast=False):
        """
        Runs the LSD algorithm on a single BGR frame.
        - fast_mode: If True, skips complex mathematical refinements to run smoothly in real-time.
        - enhance_contrast: If True, applies Histogram Equalization (useful for bad webcam lighting).
        """
        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Image Enhancement (Chapter 3)
        if enhance_contrast:
            img_in = cv2.equalizeHist(img_in)

        # 1. Image Scaling & Gaussian Blur (Anti-aliasing)
        if self.scale != 1.0:
            sigma = self.sigma_scale / self.scale if self.scale < 1.0 else self.sigma_scale
            ksize = int(math.ceil(sigma * math.sqrt(2.0 * 3.0 * math.log(10.0)))) 
            ksize = 1 + 2 * ksize
            img_blurred = cv2.GaussianBlur(img_in.astype(float), (ksize, ksize), sigma)
            self.img = cv2.resize(img_blurred, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        else:
            self.img = img_in.astype(float)
            
        self.height, self.width = self.img.shape
        self.status = np.full((self.height, self.width), NOTUSED, dtype=np.uint8)
        
        # 2. Gradient Computation (2x2 mask)
        modgrad, angles = self.compute_gradients()
        
        # 3. Pseudo-Ordering (Linear Time Sort)
        sorted_pixels = self.pseudo_ordering(modgrad)
        
        # Calculate theoretical number of tests (N*M)^(5/2) * gamma
        logNT = 5.0 * (math.log10(self.width) + math.log10(self.height)) / 2.0 + math.log10(11.0)
        min_reg_size = int(-logNT / math.log10(self.p_val))
        detections = []
        
        # 4. Region Growing and Validation
        for px, py in sorted_pixels:
            if self.status[py, px] == NOTUSED and angles[py, px] != NOTDEF:
                # A. Grow region
                reg_x, reg_y, reg_angle = self.region_grow(px, py, angles, self.prec)
                if len(reg_x) < min_reg_size: continue
                
                # B. Fit rectangle to region
                rect = self.region2rect(reg_x, reg_y, modgrad, reg_angle, self.prec, self.p_val)
                if rect is None: continue
                
                # C. Check density & Refinement
                dist_len = math.sqrt((rect['x2']-rect['x1'])**2 + (rect['y2']-rect['y1'])**2)
                density = len(reg_x) / (max(dist_len * rect['width'], 1.0))
                
                if fast_mode:
                    if density < self.density_th:
                        continue # Skip refinement in fast mode
                    # Direct NFA calculation
                    pts, k = self.get_aligned_points(rect, angles)
                    log_nfa_val = self.nfa(pts, k, rect['p'], logNT)
                    final_rect = rect
                else:
                    # Precise Mode (Applies mathematical refinements like the original paper)
                    ok, rect, reg_x, reg_y = self.refine(reg_x, reg_y, modgrad, reg_angle, self.prec, self.p_val, rect, angles)
                    if not ok: continue
                    # D. NFA Optimization
                    log_nfa_val, final_rect = self.rect_improve(rect, angles, logNT)
                
                # E. Final Validation (Threshold)
                if log_nfa_val > LOG_EPS:
                    # Restore coordinates to original scale
                    if self.scale != 1.0:
                        final_rect['x1'] = (final_rect['x1'] + 0.5) / self.scale
                        final_rect['y1'] = (final_rect['y1'] + 0.5) / self.scale
                        final_rect['x2'] = (final_rect['x2'] + 0.5) / self.scale
                        final_rect['y2'] = (final_rect['y2'] + 0.5) / self.scale
                    detections.append(final_rect)
                    
        return detections

    def compute_gradients(self):
        img = self.img
        A = img[:-1, :-1]
        B = img[:-1, 1:]
        C = img[1:, :-1]
        D = img[1:, 1:]
        
        gx = (B + D) - (A + C)
        gy = (C + D) - (A + B)
        norm2 = gx**2 + gy**2
        modgrad = np.sqrt(norm2 / 4.0) 
        
        angles = np.full(modgrad.shape, NOTDEF)
        mask = modgrad > self.rho
        angles[mask] = np.arctan2(gx[mask], -gy[mask])
        
        modgrad_full = np.zeros((self.height, self.width))
        angles_full = np.full((self.height, self.width), NOTDEF)
        modgrad_full[:-1, :-1] = modgrad
        angles_full[:-1, :-1] = angles
        
        return modgrad_full, angles_full

    def pseudo_ordering(self, modgrad):
        max_grad = np.max(modgrad)
        if max_grad == 0: return []
        bins = [[] for _ in range(self.n_bins)]
        
        flattened_grad = modgrad.flatten()
        valid_indices = np.where(flattened_grad > self.rho)[0]
        
        for idx in valid_indices:
            val = flattened_grad[idx]
            bin_idx = int(val * self.n_bins / max_grad)
            if bin_idx >= self.n_bins: bin_idx = self.n_bins - 1
            y, x = divmod(idx, self.width)
            bins[bin_idx].append((x, y))
            
        ordered_pixels = []
        for b in reversed(bins):
            ordered_pixels.extend(b)
        return ordered_pixels

    def nfa(self, n, k, p, logNT):
        if n == 0 or k == 0 or k > n or p >= 1.0: return -logNT
        log_prob_tail_10 = binom.logsf(k-1, n, p) / M_LN10
        return -(logNT + log_prob_tail_10)

    def region_grow(self, x, y, angles, prec):
        reg_x, reg_y = [x], [y]
        self.status[y, x] = USED 
        reg_angle = angles[y, x]
        sum_dx, sum_dy = math.cos(reg_angle), math.sin(reg_angle)
        
        idx = 0
        while idx < len(reg_x):
            curr_x, curr_y = reg_x[idx], reg_y[idx]
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.status[ny, nx] == NOTUSED and angles[ny, nx] != NOTDEF:
                            diff = angles[ny, nx] - reg_angle
                            while diff <= -M_PI: diff += 2*M_PI
                            while diff > M_PI: diff -= 2*M_PI
                            if abs(diff) <= prec:
                                self.status[ny, nx] = USED
                                reg_x.append(nx); reg_y.append(ny)
                                sum_dx += math.cos(angles[ny, nx])
                                sum_dy += math.sin(angles[ny, nx])
                                reg_angle = math.atan2(sum_dy, sum_dx)
            idx += 1
        return np.array(reg_x), np.array(reg_y), reg_angle

    def region2rect(self, reg_x, reg_y, modgrad, reg_angle, prec, p):
        weights = modgrad[reg_y, reg_x]
        sum_w = np.sum(weights)
        if sum_w <= 0: return None
        
        cx = np.sum(reg_x * weights) / sum_w
        cy = np.sum(reg_y * weights) / sum_w
        dx_arr, dy_arr = reg_x - cx, reg_y - cy
        
        ixx = np.sum(weights * dy_arr**2)
        iyy = np.sum(weights * dx_arr**2)
        ixy = -np.sum(weights * dx_arr * dy_arr)
        
        lambda_val = 0.5 * (ixx + iyy - math.sqrt((ixx - iyy)**2 + 4.0 * ixy * ixy))
        theta = math.atan2(lambda_val - ixx, ixy) if abs(ixx) > abs(iyy) else math.atan2(ixy, lambda_val - iyy)
            
        diff = theta - reg_angle
        while diff <= -M_PI: diff += 2*M_PI
        while diff > M_PI: diff -= 2*M_PI
        if abs(diff) > prec: theta += M_PI
        
        dx, dy = math.cos(theta), math.sin(theta)
        l = dx_arr * dx + dy_arr * dy
        w = -dx_arr * dy + dy_arr * dx
        
        l_min, l_max = np.min(l), np.max(l)
        w_min, w_max = np.min(w), np.max(w)
        
        return {
            'x1': cx + l_min * dx, 'y1': cy + l_min * dy,
            'x2': cx + l_max * dx, 'y2': cy + l_max * dy,
            'width': max(1.0, w_max - w_min),
            'x': cx, 'y': cy, 'theta': theta, 'dx': dx, 'dy': dy,
            'prec': prec, 'p': p
        }

    def get_aligned_points(self, rect, angles):
        r_half_w = rect['width'] / 2.0
        diag = math.sqrt((rect['x2']-rect['x1'])**2 + (rect['y2']-rect['y1'])**2) + rect['width']
        
        min_x = max(0, int(rect['x'] - diag/2 - 1))
        max_x = min(self.width, int(rect['x'] + diag/2 + 2))
        min_y = max(0, int(rect['y'] - diag/2 - 1))
        max_y = min(self.height, int(rect['y'] + diag/2 + 2))
        
        y_grid, x_grid = np.mgrid[min_y:max_y, min_x:max_x]
        tx, ty = x_grid - rect['x'], y_grid - rect['y']
        dx, dy = rect['dx'], rect['dy']
        
        l_pos = tx * dx + ty * dy
        w_pos = -tx * dy + ty * dx
        
        l_min = (rect['x1'] - rect['x']) * dx + (rect['y1'] - rect['y']) * dy
        l_max = (rect['x2'] - rect['x']) * dx + (rect['y2'] - rect['y']) * dy
        
        mask = (l_pos >= l_min) & (l_pos <= l_max) & (np.abs(w_pos) <= r_half_w)
        valid_y, valid_x = y_grid[mask], x_grid[mask]
        
        if len(valid_x) == 0: return 0, 0
        
        pt_angles = angles[valid_y, valid_x]
        valid_angles = pt_angles[pt_angles != NOTDEF]
        
        diff = np.mod(valid_angles - rect['theta'] + M_PI, 2*M_PI) - M_PI
        return len(valid_x), np.sum(np.abs(diff) <= rect['prec'])

    # ==========================================
    # PRECISE REFINEMENT METHODS (For Static Mode)
    # ==========================================
    def refine(self, reg_x, reg_y, modgrad, reg_angle, prec, p, rect, angles):
        dist_len = math.sqrt((rect['x2']-rect['x1'])**2 + (rect['y2']-rect['y1'])**2)
        density = len(reg_x) / max((dist_len * rect['width']), 1.0)
        
        if density >= self.density_th:
            return True, rect, reg_x, reg_y
        
        # Strategy 1: Reduce angular tolerance
        xc, yc = reg_x[0], reg_y[0] 
        ang_c = angles[yc, xc]
        dist_sq = (reg_x - xc)**2 + (reg_y - yc)**2
        radius_sq = rect['width']**2
        mask_near = dist_sq < radius_sq
        near_x = reg_x[mask_near]
        near_y = reg_y[mask_near]
        
        self.status[reg_y, reg_x] = NOTUSED
        
        if len(near_x) > 0:
            near_angles = angles[near_y, near_x]
            valid_a = near_angles[near_angles != NOTDEF]
            if len(valid_a) > 0:
                diffs = valid_a - ang_c
                diffs = np.mod(diffs + M_PI, 2*M_PI) - M_PI
                mean_angle = np.mean(diffs)
                tau = 2.0 * np.sqrt( np.mean(diffs**2) - mean_angle**2 ) 
                
                new_reg_x, new_reg_y, new_reg_angle = self.region_grow(xc, yc, angles, tau)
                if len(new_reg_x) >= 2:
                    new_rect = self.region2rect(new_reg_x, new_reg_y, modgrad, new_reg_angle, prec, p)
                    if new_rect:
                        dist_len_n = math.sqrt((new_rect['x2']-new_rect['x1'])**2 + (new_rect['y2']-new_rect['y1'])**2)
                        new_dens = len(new_reg_x) / max((dist_len_n * new_rect['width']), 1.0)
                        if new_dens >= self.density_th:
                            return True, new_rect, new_reg_x, new_reg_y
                        return self.reduce_region_radius(new_reg_x, new_reg_y, modgrad, new_reg_angle, prec, p, new_rect, new_dens)

        self.status[reg_y, reg_x] = USED
        return self.reduce_region_radius(reg_x, reg_y, modgrad, reg_angle, prec, p, rect, density)

    def reduce_region_radius(self, reg_x, reg_y, modgrad, reg_angle, prec, p, rect, curr_density):
        xc, yc = rect['x'], rect['y']
        rad1 = math.sqrt((rect['x1']-xc)**2 + (rect['y1']-yc)**2)
        rad2 = math.sqrt((rect['x2']-xc)**2 + (rect['y2']-yc)**2)
        rad = max(rad1, rad2)
        density = curr_density
        new_reg_x, new_reg_y = reg_x, reg_y
        
        while density < self.density_th:
            rad *= 0.75 
            dist_sq = (new_reg_x - xc)**2 + (new_reg_y - yc)**2
            mask = dist_sq <= rad**2
            
            rejected_x = new_reg_x[~mask]
            rejected_y = new_reg_y[~mask]
            self.status[rejected_y, rejected_x] = NOTUSED
            
            new_reg_x = new_reg_x[mask]
            new_reg_y = new_reg_y[mask]
            if len(new_reg_x) < 2: return False, None, None, None
            
            rect = self.region2rect(new_reg_x, new_reg_y, modgrad, reg_angle, prec, p)
            if rect is None: return False, None, None, None
            
            dist_len = math.sqrt((rect['x2']-rect['x1'])**2 + (rect['y2']-rect['y1'])**2)
            density = len(new_reg_x) / max((dist_len * rect['width']), 1.0)
            
        return True, rect, new_reg_x, new_reg_y

    def rect_improve(self, rect, angles, logNT):
        pts, k = self.get_aligned_points(rect, angles)
        log_nfa = self.nfa(pts, k, rect['p'], logNT)
        
        if log_nfa > LOG_EPS: return log_nfa, rect
        
        best_log_nfa = log_nfa
        best_rect = rect.copy()
        
        # 1. Try finer precision
        temp_rect = rect.copy()
        for _ in range(5):
            temp_rect['p'] /= 2.0
            temp_rect['prec'] = temp_rect['p'] * M_PI
            pts, k = self.get_aligned_points(temp_rect, angles)
            new_nfa = self.nfa(pts, k, temp_rect['p'], logNT)
            if new_nfa > best_log_nfa:
                best_log_nfa = new_nfa
                best_rect = temp_rect.copy()
        if best_log_nfa > LOG_EPS: return best_log_nfa, best_rect
        
        # 2. Try to reduce width
        temp_rect = best_rect.copy()
        delta = 0.5
        for _ in range(5):
            if temp_rect['width'] - delta >= 0.5:
                temp_rect['width'] -= delta
                pts, k = self.get_aligned_points(temp_rect, angles)
                new_nfa = self.nfa(pts, k, temp_rect['p'], logNT)
                if new_nfa > best_log_nfa:
                    best_log_nfa = new_nfa
                    best_rect = temp_rect.copy()
        if best_log_nfa > LOG_EPS: return best_log_nfa, best_rect
        
        # 3. Try lateral shifts (Side 1)
        temp_rect = best_rect.copy()
        delta_2 = delta / 2.0
        for _ in range(5):
            if temp_rect['width'] - delta >= 0.5:
                shift_x = -temp_rect['dy'] * delta_2
                shift_y =  temp_rect['dx'] * delta_2
                temp_rect['x1'] += shift_x; temp_rect['y1'] += shift_y
                temp_rect['x2'] += shift_x; temp_rect['y2'] += shift_y
                temp_rect['x'] += shift_x; temp_rect['y'] += shift_y
                temp_rect['width'] -= delta
                pts, k = self.get_aligned_points(temp_rect, angles)
                new_nfa = self.nfa(pts, k, temp_rect['p'], logNT)
                if new_nfa > best_log_nfa:
                    best_log_nfa = new_nfa
                    best_rect = temp_rect.copy()
        if best_log_nfa > LOG_EPS: return best_log_nfa, best_rect

        # 4. Try lateral shifts (Side 2)
        temp_rect = best_rect.copy()
        for _ in range(5):
            if temp_rect['width'] - delta >= 0.5:
                shift_x = -temp_rect['dy'] * delta_2
                shift_y =  temp_rect['dx'] * delta_2
                temp_rect['x1'] -= shift_x; temp_rect['y1'] -= shift_y
                temp_rect['x2'] -= shift_x; temp_rect['y2'] -= shift_y
                temp_rect['x'] -= shift_x; temp_rect['y'] -= shift_y
                temp_rect['width'] -= delta
                pts, k = self.get_aligned_points(temp_rect, angles)
                new_nfa = self.nfa(pts, k, temp_rect['p'], logNT)
                if new_nfa > best_log_nfa:
                    best_log_nfa = new_nfa
                    best_rect = temp_rect.copy()
        if best_log_nfa > LOG_EPS: return best_log_nfa, best_rect

        # 5. Try even finer precision
        temp_rect = best_rect.copy()
        for _ in range(5):
            temp_rect['p'] /= 2.0
            temp_rect['prec'] = temp_rect['p'] * M_PI
            pts, k = self.get_aligned_points(temp_rect, angles)
            new_nfa = self.nfa(pts, k, temp_rect['p'], logNT)
            if new_nfa > best_log_nfa:
                best_log_nfa = new_nfa
                best_rect = temp_rect.copy()
                
        return best_log_nfa, best_rect


# --- CAMERA CALIBRATION (Chapter 6) ---
def calibrate_frame(frame, active):
    """
    Simulates camera calibration by correcting radial (barrel) distortion.
    """
    if not active:
        return frame
    
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.2, 0.1, 0, 0], dtype=np.float32) 
    undistorted = cv2.undistort(frame, K, D)
    return undistorted

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Initializing LSD Algorithm...")
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    USE_WEBCAM = False                    # Set to True for live video, False for static image
    IMAGE_PATH = "images/test_image.png"      # Path to your static image (change name appropriately)
    # ==========================================
    
    lsd_detector = LSD(scale=0.8)
    calibration_active = False

    if USE_WEBCAM:
        # --- WEBCAM MODE (Fast, Enhanced Contrast) ---
        print("Starting Webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("\nCONTROLS:")
        print("- Press 'c' to Toggle Camera Calibration")
        print("- Press 'q' to Quit")

        cv2.namedWindow("LSD - Live Webcam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LSD - Live Webcam", 800, 600)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame = calibrate_frame(frame, calibration_active)
            
            # fast_mode=True skips heavy math. enhance_contrast=True balances bad lighting.
            lines = lsd_detector.run(processed_frame, fast_mode=True, enhance_contrast=True)
            
            display_frame = processed_frame.copy()
            for d in lines:
                x1, y1 = int(round(d['x1'])), int(round(d['y1']))
                x2, y2 = int(round(d['x2'])), int(round(d['y2']))
                # Drawn in BLUE (BGR format: 255, 0, 0)
                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
                
            status_text = "Lens Calibration: ON" if calibration_active else "Lens Calibration: OFF"
            color = (255, 0, 0) if calibration_active else (0, 0, 255) # Blue text when active
            cv2.putText(display_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(display_frame, f"Segments: {len(lines)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("LSD - Live Webcam", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibration_active = not calibration_active

        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # --- STATIC IMAGE MODE (Precise, Original Contrast) ---
        print(f"Loading static image from: {IMAGE_PATH}")
        frame = cv2.imread(IMAGE_PATH)
        
        if frame is None:
            print(f"\n[!] ERROR: Could not load image at '{IMAGE_PATH}'.")
            print("Please make sure you create an 'images' folder and put the image inside it.")
        else:
            print("Processing image with HIGH PRECISION MODE, please wait...")
            
            # fast_mode=False uses mathematical Refinement. enhance_contrast=False preserves real gradients.
            lines = lsd_detector.run(frame, fast_mode=False, enhance_contrast=False)
            print(f"Done! Found {len(lines)} segments.")
            
            display_frame = frame.copy()
            for d in lines:
                x1, y1 = int(round(d['x1'])), int(round(d['y1']))
                x2, y2 = int(round(d['x2'])), int(round(d['y2']))
                # Drawn in BLUE (BGR format: 255, 0, 0)
                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.putText(display_frame, f"Segments: {len(lines)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            cv2.namedWindow("LSD - Static Image (High Precision)", cv2.WINDOW_NORMAL)
            cv2.imshow("LSD - Static Image (High Precision)", display_frame)
            
            print("\nPress ANY KEY on the image window to close it.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()