from .core import cpu, Signal
import numpy as np
import matplotlib.pyplot as plt
from DensityPlot import density2d


def scatterplot(signal, interval=1, is_density=False, size=1):
    with cpu(signal):

        try:
            samples = np.atleast_2d(signal.samples[:, ::interval])
        except AttributeError:
            samples = np.copy(signal)

        with plt.style.context(['ieee', 'science', 'grid', 'no-latex']):
            fig, axes = plt.subplots(1, samples.shape[0])
            axes = np.atleast_2d(axes)[0]
            pol = 0
            xlim = [samples[pol].real.min() - samples[pol].real.min() / 3,
                    samples[pol].real.max() + samples[pol].real.max() / 3]
            ylim = [samples[pol].imag.min() - samples[pol].imag.min() / 3
                , samples[pol].imag.max() + samples[pol].imag.max() / 3]
            for ax in axes:

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if not is_density:
                    ax.scatter(samples[pol].real, samples[pol].imag, c='b', s=size)
                else:
                    density2d(x=samples[pol].real, y=samples[pol].imag, bins=500, ax=ax, s=size)
                pol += 1
                ax.set_aspect(1)
            # viz = visdom.Visdom()
            # viz.matplot(fig)
            plt.tight_layout()
            plt.show()


def snr_meter(signal: Signal):
    with cpu(signal):
        signal.normalize()
        assert signal.shape == signal.symbol.shape
        noise = signal[:, 1024:-1024] - signal.symbol[:, 1024:-1024]
        noise_power = np.sum(np.mean(np.abs(noise[:]) ** 2, axis=-1))
        return 10 * np.log10((2 - noise_power) / noise_power)
