from typing import Union

import numpy as np
from scipy.constants import c, h
from .core import backend_manager, WdmSignal, QamSignal


class Fiber:

    def __init__(self,
                 D: float,
                 alpha_db: float,
                 length: float,
                 wavelength_nm: float,
                 slope: float):
        self.D = D
        self.alpha_db = alpha_db
        self.length = length
        self.reference_wavelength_nm = wavelength_nm
        self.slope = slope

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength_nm * 1e-12) ** 2 \
               / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length_m):
        '''
        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * \
             (1 / wave_length_m - 1 / (self.reference_wavelength_nm * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength_nm * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength_nm * 1e-12 * self.D + (
                self.reference_wavelength_nm * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        '''
        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length

    @property
    def alphalin(self):
        alphalin = self.alpha_db / (10 * np.log10(np.exp(1)))
        return alphalin


class NonlinearFiber(Fiber):

    def __init__(self,
                 D: float = 16.7,
                 alpha_db: float = 0.2,
                 length: float = 80,
                 wavelength_nm: float = 1550,
                 slop: float = 0,
                 gamma: float = 1.3,
                 step_length: float = 20 / 1000
                 ):

        super(NonlinearFiber, self).__init__(D, alpha_db, length, wavelength_nm, slop)
        self.gamma = gamma
        self.step_length = step_length

    @property
    def step_length_eff(self):
        return self.leff(self.step_length)

    def propagation(self, signal: Union[QamSignal, WdmSignal]) -> Union[QamSignal, WdmSignal]:
        backend, ith_device, selected_device = backend_manager(signal)
        with selected_device(ith_device):
            nstep = self.length / self.step_length
            nstep = int(backend.floor(nstep))
            freq = backend.fft.fftfreq(signal.shape[1], 1 / signal.fs)
            omeg = 2 * backend.pi * freq
            self.D = -1j / 2 * self.beta2(c / signal.center_freq) * omeg ** 2
            N = 8 / 9 * 1j * self.gamma
            atten = -self.alphalin / 2
            last_step = self.length - self.step_length * nstep

            signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.step_length / 2)
            signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1])
            signal[0] = signal[0] * backend.exp(atten * self.step_length)
            signal[1] = signal[1] * backend.exp(atten * self.step_length)

            for _ in range(nstep - 1):
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.step_length)

                signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1])
                signal[0] = signal[0] * backend.exp(atten * self.step_length)
                signal[1] = signal[1] * backend.exp(atten * self.step_length)

            signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.step_length / 2)

            if last_step:
                last_step_eff = (1 - backend.exp(-self.alphalin * last_step)) / self.alphalin
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], last_step / 2)
                signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1], last_step_eff)
                signal[0] = signal[0] * backend.exp(atten * last_step)
                signal[1] = signal[1] * backend.exp(atten * last_step)
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], last_step / 2)

            return signal

    def nonlinear_prop(self, backend, N, time_x, time_y, step_length=None):
        if step_length is None:
            time_x = time_x * backend.exp(
                N * self.step_length_eff * (backend.abs(time_x) ** 2 + backend.abs(
                    time_y) ** 2))
            time_y = time_y * backend.exp(
                N * self.step_length_eff * (backend.abs(time_x) ** 2 + backend.abs(time_y) ** 2))
        else:
            time_x = time_x * backend.exp(
                N * step_length * (backend.abs(time_x) ** 2 + backend.abs(
                    time_y) ** 2))
            time_y = time_y * backend.exp(
                N * step_length * (backend.abs(time_x) ** 2 + backend.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop(self, backend, timex, timey, length):
        D = self.D
        freq_x = backend.fft.fft(timex)
        freq_y = backend.fft.fft(timey)

        freq_x = freq_x * backend.exp(D * length)
        freq_y = freq_y * backend.exp(D * length)

        time_x = backend.fft.ifft(freq_x)
        time_y = backend.fft.ifft(freq_y)
        return time_x, time_y


class EDFA:

    def __init__(self, gain, nf):
        self.gain = gain

        self.nf = nf

    def calc_noise_power(self, wavelength_m, fs):
        '''
        One pol
        '''
        ase_psd = (h * c / wavelength_m) * (self.gain_linear * 10 ** (self.nf / 10) - 1) / 2
        noise_power = ase_psd * fs
        return noise_power

    @property
    def gain_linear(self):
        return 10 ** (self.gain / 10)

    def noise_sequence(self, signal):
        noise_power = self.calc_noise_power(c / signal.center_freq, signal.fs)

        backend, ith_device, selected_device = backend_manager(signal)
        with selected_device(ith_device):
            noise_sequence = backend.sqrt(noise_power / 2) * (
                    backend.random.randn(*signal.shape) + 1j * backend.random.randn(*signal.shape))

        return noise_sequence


class ConstantGainEDFA(EDFA):

    def __init__(self, gain, nf):
        super(ConstantGainEDFA, self).__init__(gain, nf)

    def propagation(self, signal):
        backend, ith_device, selected_device = backend_manager(signal)

        with selected_device(ith_device):
            noise_sequence = self.noise_sequence(signal)
            noise_power = self.calc_noise_power(c / signal.center_freq, signal.fs)
            psd = noise_power / signal.fs
            signal[:] = signal[:] * backend.sqrt(self.gain_linear)
            signal[:] = signal[:] + noise_sequence
        return signal


class WSS:

    def __init__(self, center_frequency, bandwidth, otf):
        '''
            Hz

        '''

        self.center_freq = center_frequency
        self.bandwidth = bandwidth
        self.otf = otf

        self.H = None

    def get_transfer_function(self, freq_vector, device, backend):

        if 'cuda' in device:
            from cupyx.scipy.special import erf
        else:
            from scipy.special import erf
        delta = self.otf / 2 / backend.sqrt(2 * backend.log(2))

        H = 0.5 * delta * backend.sqrt(2 * backend.pi) * (
                erf((self.bandwidth / 2 - (freq_vector - self.center_freq)) / backend.sqrt(2) / delta) - erf(
            (-self.bandwidth / 2 - (freq_vector - self.center_freq)) / backend.sqrt(2) / delta))

        H = H / backend.max(H)

        self.H = H

    def propgation(self, signal):
        backend, ith_device, selected_device = backend_manager(signal)
        with selected_device(ith_device):
            freq = backend.fft.fftfreq(signal.shape[1], 1 / signal.fs)
            self.get_transfer_function(freq, signal.device, backend)
            signal[:] = backend.fft.ifft(backend.fft.fft(signal[:], axis=-1) * self.H)
            return signal
