from typing import List

import numpy as np
from contextlib import contextmanager
from .constl import cal_symbols_qam, cal_scaling_factor_qam


###################################Functions for selected proper devices###################################

@contextmanager
def cpu(signal):
    original_device = signal.device
    signal.to('cpu')
    yield signal
    signal.to(original_device)


@contextmanager
def np_device(backend):
    yield backend


def backend_manager(signal):
    if signal.device == "cpu":
        return np,None,np_device
    if "cuda" in signal.device:
        import cupy as cp
        return cp,int(signal.device.split(":")[-1]),cp.cuda.Device
###########################################################################################################

class Signal:
    def __init__(self,baudrate,sps,pol_number,center_freq):
        self.baudrate = baudrate
        self.sps = sps
        self.center_freq = center_freq
        self.pol_number = pol_number

        self.samples = None
        self.symbol = None
        self.bit_sequence = None
        self.device = "cpu"

    def to(self,target_devcie):
        if self.device == target_devcie:
            return self
        elif target_devcie == "cpu":
            import cupy as cp
            self.samples = cp.asnumpy(self.samples)
            self.device = target_devcie
        else:
            import cupy as cp
            with cp.cuda.Device(int(target_devcie.split(":")[-1])):
                self.samples = cp.asarray(self.samples)

            self.device = target_devcie

        return self

    def __setitem__(self, key, value):
        self.samples[key] = value

    def __getitem__(self, item):
        return self.samples[item]

    @property
    def fs(self):
        return self.sps * self.baudrate

    @property
    def power_watt(self):
        backend, ith_device, selected_device = backend_manager(self)
        with selected_device(ith_device):
            power = backend.sum(backend.mean(backend.abs(self[:]) ** 2, axis=-1))
        return power

    @property
    def power_dbm(self):
        backend, ith_device, selected_device = backend_manager(self)
        with selected_device(ith_device):
            return  10*backend.log10(self.power_watt*1000)

    def normalize(self):
        backend, ith_device, selected_device = backend_manager(self)
        with selected_device(ith_device):
            factor = backend.sqrt(backend.mean(backend.abs(self[:])**2,axis=-1,keepdims=True))
            self[:]/=factor
        return self

    def map(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.samples.shape


class QamSignal(Signal):

    def __init__(self,qam_order,symbol_number,baudrate,sps,pol_number,center_freq,device):
        super(QamSignal, self).__init__(baudrate,sps,pol_number,center_freq)
        self.symbol_number = symbol_number
        self.qam_order = qam_order
        self.bit_sequence = np.random.randint(0, 2, (self.pol_number, int(symbol_number*np.log2(qam_order))), dtype=bool)
        self.constl = cal_symbols_qam(self.qam_order) / np.sqrt(cal_scaling_factor_qam(self.qam_order))

        self.map()
        self.samples = np.zeros(shape=(self.pol_number, self.symbol_number * self.sps), dtype=np.complex128)
        self.samples[:, ::self.sps] = self.symbol
        self.samples = np.atleast_2d(self.samples)
        self.symbol = np.atleast_2d(self.symbol)
        self.bit_sequence = np.atleast_2d(self.bit_sequence)
        self.to(device)


    def map(self):
        from .constl import generate_mapping, map
        _, encoding = generate_mapping(self.qam_order)
        self.symbol = map(self.bit_sequence, encoding=encoding, M=self.qam_order)


class WdmSignal:

    def __init__(self,samples,symbols:List[np.ndarray],
                 absolute_freq:List[float],center_freq:float,
                 device:str,fs:float):

        self.samples = samples
        self.symbols = symbols
        self.freq = np.array(absolute_freq)
        self.center_freq = center_freq

        self.relative_freq = self.freq - center_freq
        self.device = device
        self.fs = fs
        self.to(device)

    def to(self,device):
        if self.device == device:
            return self
        elif device=="cpu":
            import cupy as cp
            self.samples = cp.asnumpy(self.samples)
            self.device = device
        elif "cuda" in device:
            import cupy as cp
            with cp.cuda.Device(int(device.split(":")[-1])):
                self.samples = cp.asarray(self.samples)

            self.device = device
        return self

    @property
    def power_watt(self):
        backend, ith_device, selected_device = backend_manager(self)
        with selected_device(ith_device):
            power = backend.sum(backend.mean(backend.abs(self[:]) ** 2, axis=-1))
        return power

    @property
    def power_dbm(self):
        backend, ith_device, selected_device = backend_manager(self)
        with selected_device(ith_device):
            return 10 * backend.log10(self.power_watt() * 1000)

    @property
    def shape(self):
        return self.samples.shape
