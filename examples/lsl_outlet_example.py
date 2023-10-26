"""
In this example, we create a System comprising a simple graph from a
EEGSynth to an LSL Outlet.
"""
from typing import Tuple

import ezmsg.core as ez
from ezmsg.sigproc.synth import EEGSynth, EEGSynthSettings
from ezmsg.lsl.units import LSLOutletUnit, LSLOutletSettings


class LSLDemoSystemSettings(ez.Settings):
    fs: float = 1000.0  # Hz
    n_time: int = 100
    alpha_freq: float = 10.5  # Hz
    n_ch: int = 128
    stream_name: str = "ezmsg-EEGSynth"
    stream_type: str = "EEG"


class LSLDemoSystem(ez.Collection):
    SETTINGS: LSLDemoSystemSettings

    EEG = EEGSynth()
    OUTLET = LSLOutletUnit()

    def configure(self) -> None:
        self.EEG.apply_settings(
            EEGSynthSettings(
                fs=self.SETTINGS.fs,
                n_time=self.SETTINGS.n_time,
                alpha_freq=self.SETTINGS.alpha_freq,
                n_ch=self.SETTINGS.n_ch,
            )
        )
        self.OUTLET.apply_settings(
            LSLOutletSettings(
                stream_name=self.SETTINGS.stream_name,
                stream_type=self.SETTINGS.stream_type,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return ((self.EEG.OUTPUT_SIGNAL, self.OUTLET.INPUT_SIGNAL),)

    def process_components(self) -> Tuple[ez.Component, ...]:
        return self.EEG, self.OUTLET


if __name__ == "__main__":
    # Run the websocket system
    system = LSLDemoSystem()
    system.apply_settings(
        LSLDemoSystemSettings(stream_name="ezmsg-EEGSynth", stream_type="EEG")
    )
    ez.run(SYSTEM=system)
