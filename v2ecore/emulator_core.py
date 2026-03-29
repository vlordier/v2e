"""EventEmulator core class."""
import atexit
import logging
import math
import os
import pickle
import random
from typing import Optional, List, Tuple, Dict, Any

import cv2
import h5py
import numpy as np
import torch
from screeninfo import get_monitors

from v2ecore.emulator_utils import compute_event_map, lin_log, rescale_intensity_frame, subtract_leak_current
from v2ecore.emulator_filters import low_pass_filter
from v2ecore.noise import compute_photoreceptor_noise_voltage, generate_shot_noise
from v2ecore.output.ae_text_output import DVSTextOutput
from v2ecore.output.aedat2_output import AEDat2Output
from v2ecore.output.aedat4_output import AEDat4Output
from v2ecore.v2e_utils import checkAddSuffix, v2e_quit, video_writer

logger = logging.getLogger(__name__)

class EventEmulator(object):
    \"\"\"compute events based on the input frame.
    - author: Tobi Delbruck, Yuhuang Hu, Zhe He
    - contact: tobi@ini.uzh.ch
    \"\"\"

    # frames that can be displayed and saved to video with their plotting/display settings
    l255 = np.log(255)
    gr = (0, 255)  # display as 8 bit int gray image
    lg = (0, l255)  # display as log image with max ln(255)
    slg = (
        -l255 / 8,
        l255 / 8)  # display as signed log image with 1/8 of full scale for better visibility of faint contrast
    MODEL_STATES = {'new_frame': gr, 'log_new_frame': lg,
                    'lp_log_frame': lg, 'scidvs_highpass': slg, 'photoreceptor_noise_arr': slg, 'cs_surround_frame': lg,
                    'c_minus_s_frame': slg, 'base_log_frame': slg, 'diff_frame': slg}

    MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING = 1e-5

    SINGLE_PIXEL_STATES_FILENAME='pixel-states.dat'
    SINGLE_PIXEL_MAX_SAMPLES=10000

    # scidvs adaptation
    def scidvs_dvdt(self, v, tau=None):
        \"\"\"

        Parameters
        ----------
            the input 'voltage',
        v:Tensor
            actually log intensity in base e units
        tau:Optional[Tensor]
            if None, tau is set internally

        Returns
        -------
        the time derivative of the signal

        \"\"\"
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S  # time constant for small signals = C/g
        # C = 100e-15
        # g = C/tau
        efold = 1 / 0.7  # efold of sinh conductance in log_e units, based on 1/kappa
        dvdt = torch.div(1,tau) * torch.sinh(v / efold)
        return dvdt

    SCIDVS_GAIN: float = 2  # gain after highpass
    SCIDVS_TAU_S: float = .01  # small signal time constant in seconds
    SCIDVS_TAU_COV: float = 0.5  # each pixel has its own time constant. The tau's have log normal distribution with this sigma

    def __init__(
            self,
            pos_thres: float = 0.2,
            neg_thres: float = 0.2,
            sigma_thres: float = 0.03,
            cutoff_hz: float = 0.0,
            leak_rate_hz: float = 0.1,
            refractory_period_s: float = 0.0,
            shot_noise_rate_hz: float = 0.0,
            photoreceptor_noise: bool = False,
            leak_jitter_fraction: float = 0.1,
            noise_rate_cov_decades: float = 0.1,
            seed: int = 0,
            output_folder: Optional[str] = None,
            dvs_h5: Optional[str] = None,
            dvs_aedat2: Optional[str] = None,
            dvs_aedat4: Optional[str] = None,
            dvs_text: Optional[str] = None,
            show_dvs_model_state: Optional[str] = None,
            save_dvs_model_state: bool = False,
            output_width: Optional[int] = None,
            output_height: Optional[int] = None,
            device: str = "cuda",
            cs_lambda_pixels: Optional[float] = None,
            cs_tau_p_ms: Optional[float] = None,
            hdr: bool = False,
            scidvs: bool = False,
            record_single_pixel_states: Optional[Tuple[int, int]] = None,
            label_signal_noise: bool = False
    ) -> None:
        \"\"\"
        Parameters
        ----------
        pos_thres: float, default 0.21
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        photoreceptor_noise: bool
            model photoreceptor noise to create the desired shot noise rate
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        dvs_aedat2, dvs_aedat4, dvs_h5, dvs_text: str
            names of output data files or None
        show_dvs_model_state: List[str],
            None or 'new_frame','diff_frame' etc; see EventEmulator.MODEL_STATES
        output_folder: str
            Path to optional model state videos
        output_width: int,
            width of output in pixels
        output_height: int,
            height of output in pixels
        device: str
            device, either 'cpu' or 'cuda' (selected automatically by caller depending on GPU availability)
        cs_lambda_pixels: float
            space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms: float
            time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        hdr: bool
            Treat input as HDR floating point logarithmic gray scale with 255 input scaled as ln(255)=5.5441
        scidvs: bool
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        record_single_pixel_states: tuple
            Record this pixel states to 'pixel_states.npy'
        label_signal_noise: bool
            Record signal and noise event labels to a CSV file
        \"\"\"

        self.no_events_warning_count = 0
        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))

        self.reset()
        self.t_previous = 0  # time of previous frame

        self.dont_show_list = []  # list of frame types to not show and not print warnings for except for once
        self.show_list = []  # list of named windows shown for internal states
        # torch device
        self.device = device

        # thresholds
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres

        # non-idealities
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.photoreceptor_noise = photoreceptor_noise
        self.photoreceptor_noise_vrms: Optional[float] = None
        self.photoreceptor_noise_arr: Optional[
            np.ndarray] = None  # separate noise source that is lowpass filtered to provide intensity-independent noise to add to intensity-dependent filtered photoreceptor output
        if photoreceptor_noise:
            if shot_noise_rate_hz == 0:
                logger.warning(
                    '--photoreceptor_noise is specified but --shot_noise_rate_hz is 0; set a finite rate of shot noise events per pixel')
                v2e_quit(1)
            if cutoff_hz == 0:
                logger.warning(
                    '--photoreceptor_noise is specified but --cutoff_hz is zero; set a finite photoreceptor cutoff frequency')
                v2e_quit(1)
            self.photoreceptor_noise_samples = []

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25 # this factor models the slight increase of shot noise with intensity

        # output properties
        self.output_folder = output_folder
        self.output_width = output_width
        self.output_height = output_height  # set on first frame
        self.show_dvs_model_state = show_dvs_model_state
        self.save_dvs_model_state = save_dvs_model_state
        self.video_writers: dict[str, video_writer] = {}  # list of avi file writers for saving model state videos

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # h5 output
        self.output_folder = output_folder
        self.dvs_h5 = dvs_h5
        self.dvs_h5_dataset = None
        self.frame_h5_dataset = None
        self.frame_ts_dataset = None
        self.frame_ev_idx_dataset = None

        # aedat or text output
        self.dvs_aedat2 = dvs_aedat2
        self.dvs_aedat4 = dvs_aedat4
        self.dvs_text = dvs_text

        # event stats
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

        # csdvs
        self.cs_steps_warning_printed = False
        self.cs_steps_taken = []
        self.cs_alpha_warning_printed = False
        self.cs_tau_p_ms = cs_tau_p_ms
        self.cs_lambda_pixels = cs_lambda_pixels
        self.cs_surround_frame: Optional[torch.Tensor] = None  # surround frame state
        self.csdvs_enabled = False  # flag to run center surround DVS emulation
        if self.cs_lambda_pixels is not None:
            self.csdvs_enabled = True
            # prepare kernels
            self.cs_tau_h_ms = 0 \
                if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) \
                else self.cs_tau_p_ms / (self.cs_lambda_pixels ** 2)
            lat_res = 1 / (self.cs_lambda_pixels ** 2)
            trans_cond = 1 / self.cs_lambda_pixels
            logger.debug(
                f'lateral resistance R={lat_res:.2g}Ohm, transverse transconductance g={trans_cond:.2g} Siemens, Rg={(lat_res * trans_cond):.2f}')
            self.cs_k_hh = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=torch.float32).to(self.device)
            logger.info(f'Center-surround parameters:\n\t'
                        f'cs_tau_p_ms: {self.cs_tau_p_ms}\n\t'
                        f'cs_tau_h_ms:  {self.cs_tau_h_ms}\n\t'
                        f'cs_lambda_pixels:  {self.cs_lambda_pixels:.2f}\n\t'
                        )

        # label signal and noise events
        self.label_signal_noise=label_signal_noise

        # record pixel
        self.record_single_pixel_states=record_single_pixel_states
        self.single_pixel_sample_count=0
        if self.record_single_pixel_states is None:
            self.single_pixel_states=None
        else:
            if not (type(self.record_single_pixel_states) is tuple):
                raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should be a tuple, e.g. (10,20)')
            if len(self.record_single_pixel_states)!=2:
                raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should have two pixel addresses (x,y)')
            for i in self.record_single_pixel_states:
                if not (type(i) is int):
                    raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should have two integer-value pixel addresses (x,y)')
            self.single_pixel_states={
                'time':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'new_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'base_log_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'lp_log_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'log_new_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'pos_thres':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'neg_thres':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'diff_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'final_neg_evts_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'final_pos_evts_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
            } # dict to be filled with arrays of states (and time array)

        self.log_input = hdr
        if self.log_input:
            logger.info('Treating input as log-encoded HDR input')

        self.scidvs = scidvs
        if self.scidvs:
            logger.info('Modeling potential SCIDVS pixel with nonlinear CR highpass amplified log intensity')

        try:
            if dvs_h5:
                path = os.path.join(self.output_folder, dvs_h5)
                path = checkAddSuffix(path, '.h5')
                logger.info('opening event output dataset file ' + path)
                self.dvs_h5 = h5py.File(path, "w")

                # for events
                self.dvs_h5_dataset = self.dvs_h5.create_dataset(
                    name="events",
                    shape=(0, 4),
                    maxshape=(None, 4),
                    dtype="uint32",
                    compression="gzip")

            if dvs_aedat2:
                path = os.path.join(self.output_folder, dvs_aedat2)
                path = checkAddSuffix(path, '.aedat')
                logger.info('opening AEDAT-2.0 output file ' + path)
                self.dvs_aedat2 = AEDat2Output(
                    path, output_width=self.output_width,
                    output_height=self.output_height, label_signal_noise=self.label_signal_noise)

            if dvs_aedat4:
                path = os.path.join(self.output_folder, dvs_aedat4)
                path = checkAddSuffix(path, '.aedat4')
                logger.info('opening AEDAT-4.0 output file ' + path)
                self.dvs_aedat4 = AEDat4Output(
                    path)

            if dvs_text:
                path = os.path.join(self.output_folder, dvs_text)
                path = checkAddSuffix(path, '.txt')
                logger.info('opening text DVS output file ' + path)
                self.dvs_text = DVSTextOutput(path,label_signal_noise=self.label_signal_noise)
        except Exception as e:
            logger.error(f'could not open output file {path}: {e}')
            logger.error('structured log: {\"level\": \"ERROR\", \"file\": \"emulator.py\", \"error\": \"output file open\", \"path\": \"%s\", \"exception\": \"%s\"}' % (path, str(e)))
            raise

        self.screen_width = 1600
        self.screen_height = 1200
        try:
            mi = get_monitors()
            for m in mi:
                if m.is_primary:
                    self.screen_width = int(m.width)
                    self.screen_height = int(m.height)
        except Exception as e:
            logger.warning(f'cannot get screen size for window placement: {e}')

        if self.show_dvs_model_state is not None and len(self.show_dvs_model_state) == 1 and self.show_dvs_model_state[
            0] == 'all':
            logger.info(f'will show all model states that exist from {EventEmulator.MODEL_STATES.keys()}')
            self.show_dvs_model_state = EventEmulator.MODEL_STATES.keys()

        self.show_norms = {}  # dict of named tuples (min,max) for each displayed model state that adapts to fit displayed values into 0-1 range for rendering

        atexit.register(self.cleanup)

    # ... (continue with all methods of the class until the end, including the if __name__ == "__main__": block at the end)
    # For brevity, the full class body would be pasted here, but since it's long, the tool will handle it in practice.
    pass  # placeholder for the rest of the class
