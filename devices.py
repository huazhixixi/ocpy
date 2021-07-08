from typing import Union, List

import numpy as np

from .core import Signal, WdmSignal
from .core import backend_manager

"""
backend, ith_device, selected_device = backend_manager(self)
    with selected_device(ith_device):
        power = backend.sum(backend.mean(backend.abs(self[:]) ** 2, axis=-1))
"""


def quantiz(backend, nbits: int, clipping_ratio: float, samples: np.ndarray) -> np.ndarray:
    assert samples.ndim == 1
    power = backend.mean(np.abs(samples) ** 2)
    A = 10 ** (clipping_ratio / 20) * np.sqrt(power)
    swing = 2 * A
    delta = swing / 2 ** nbits
    levels_out = backend.linspace(-A + delta / 2, A - delta / 2, 2 ** nbits)
    levels_dec = levels_out + delta / 2
    quanti_samples = levels_out[backend.digitize(samples, levels_dec[:-1], right=False)]
    return quanti_samples


def rrcos_time(backend, alpha, span, sps):
    '''
    Function:
        calculate the impulse response of the RRC
    Return:
        b,normalize the max value to 1
    '''
    assert divmod(span * sps, 2)[1] == 0
    M = span / 2
    n = backend.arange(-M * sps, M * sps + 1)
    b = backend.zeros(len(n))
    sps *= 1
    a = alpha
    Ns = sps
    for i in range(len(n)):
        if abs(1 - 16 * a ** 2 * (n[i] / Ns) ** 2) <= backend.finfo(backend.float).eps / 2:
            b[i] = 1 / 2. * ((1 + a) * backend.sin((1 + a) * backend.pi / (4. * a)) - (1 - a) * backend.cos(
                (1 - a) * backend.pi / (4. * a)) + (4 * a) / backend.pi * backend.sin((1 - a) * backend.pi / (4. * a)))
        else:
            b[i] = 4 * a / (backend.pi * (1 - 16 * a ** 2 * (n[i] / Ns) ** 2))
            b[i] = b[i] * (backend.cos((1 + a) * backend.pi * n[i] / Ns) + backend.sinc((1 - a) * n[i] / Ns) * (
                    1 - a) * backend.pi / (
                                   4. * a))
    return b / backend.sqrt(backend.sum(backend.abs(b) ** 2))


class Device:

    def __init__(self, order):
        self.order = order

    def propagation(self, signal: Signal) -> Signal:
        raise NotImplementedError

    def __call__(self, signal: Signal) -> Signal:
        return self.propagation(signal)


class Resampler(Device):

    def __init__(self, order, sps):
        super().__init__(order)
        self.sps = sps

    def propagation(self, signal: Signal) -> Signal:
        from .core import cpu
        with cpu(signal):
            from resampy import resample
            signal.samples = resample(signal[:], signal.sps, self.sps)
            signal.sps = self.sps
        return signal


class DAC(Device):

    def __init__(self, order: int, nbits: Union[float, int], clip_ratio: Union[int, float], sps: int):
        super().__init__(order)
        self.nbits = nbits
        self.clip_ratio = clip_ratio
        self.resampler = Resampler(None, sps)

    def propagation(self, signal: Signal) -> Signal:
        from .core import cpu
        # Quantization
        with cpu(signal):
            signal = self.resampler(signal)
            for pol in signal.pol_number:
                real_quantiz = quantiz(np, self.nbits, self.clip_ratio, signal[pol].real)
                imag_quantiz = quantiz(np, self.nbits, self.clip_ratio, signal[pol].imag)
                signal[pol] = real_quantiz + 1j * imag_quantiz
        # ZOH
        return signal


class ADC(Device):

    def __init__(self, order: int, nbits: Union[float, int], clip_ratio: Union[int, float], sps: Union[int, float] = 2):
        super().__init__(order)
        self.nbits = nbits
        self.clip_ratio = clip_ratio
        self.resampler = Resampler(order=None, sps=sps)

    def propagation(self, signal: Signal) -> Signal:
        from .core import cpu

        with cpu(signal):
            if signal.sps % self.resampler.sps:
                signal.samples = signal.samples[:, ::int(signal.sps / self.resampler.sps)]
            else:
                signal = self.resampler(signal)
                for pol in signal.pol_number:
                    real_quantiz = quantiz(np, self.nbits, self.clip_ratio, signal[pol].real)
                    imag_quantiz = quantiz(np, self.nbits, self.clip_ratio, signal[pol].imag)
                    signal[pol] = real_quantiz + 1j * imag_quantiz
        return signal


class PulseShaping(Device):

    def __init__(self, order: int, roll_off: float = 0.2, ntaps: int = 1024):
        super(PulseShaping, self).__init__(order)
        self.roll_off = roll_off
        self.ntaps = ntaps
        self.h = None

    def propagation(self, signal: Signal) -> Signal:
        backend, ith_device, selected_device = backend_manager(signal)
        with selected_device(ith_device):
            self.h = rrcos_time(backend, self.roll_off, self.ntaps, signal.sps)
            delay = self.ntaps / 2 * signal.sps

            for index, row in enumerate(signal[:]):
                temp = backend.convolve(signal[index], self.h)
                temp = backend.roll(temp, -int(delay))
                signal[index] = temp[:signal.shape[1]]
        return signal


class Laser(Device):

    def __init__(self, order: int, fo: float, lw: float):
        super().__init__(order)
        self.fo = fo
        self.lw = lw

    def propagation(self, signal: Signal) -> Signal:

        backend, ith_device, selected_device = backend_manager(signal)
        with selected_device(ith_device):
            var = 2 * backend.pi * self.lw / signal.fs
            f = backend.random.normal(scale=backend.sqrt(var), size=signal.shape)
            if len(f.shape) > 1:
                f = backend.cumsum(f, axis=1)
            else:
                f = backend.cumsum(f)
            signal[:] = signal[:] * backend.exp(1j * f).astype(signal[:].dtype)
        return signal



class Mux(Device):

    def __init__(self, order, center_freq=None):
        super(Mux, self).__init__(order)
        self.center_freq = center_freq

    def propagation(self, signals: List[Signal]) -> WdmSignal:
        freq = np.array([signal.center_freq for signal in signals])

        if self.center_freq is None:
            self.center_freq = np.mean(np.max(freq) + np.min(freq))
        backend, ith_device, selected_device = backend_manager(signals[0])
        with selected_device(ith_device):

            relative_freq = freq - self.center_freq
            t = (1 / signals[0].fs) * backend.arange(signals[0].shape[1])

            wdm_sampes = 0

            for index, signal in enumerate(signals):
                wdm_sampes += signal[:] * backend.exp(1j * 2 * backend.pi * relative_freq[index] * t)

        wdm_signal = WdmSignal(wdm_sampes, [signal.symbol for signal in signals],
                               [signal.center_freq for signal in signals],
                               relative_freq, signal.device, signal.fs
                               )
        return wdm_signal


class DeMux(Device):
    pass


class IQ:
    pass


class MZM:
    pass
