import asyncio
from dataclasses import dataclass, replace
import time
from typing import AsyncGenerator

import numpy as np
import pylsl

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


# Reproduce pylsl.string2fmt but add float64 for more familiar numpy usage
string2fmt = {
    "float32": pylsl.cf_float32,
    "double64": pylsl.cf_double64,
    "float64": pylsl.cf_double64,
    "string": pylsl.cf_string,
    "int32": pylsl.cf_int32,
    "int16": pylsl.cf_int16,
    "int8": pylsl.cf_int8,
    "int64": pylsl.cf_int64,
}

fmt2npdtype = {
    pylsl.cf_double64: float,  # Prefer native type for float64
    pylsl.cf_int64: int,  # Prefer native type for int64
    pylsl.cf_float32: np.float32,
    pylsl.cf_int32: np.int32,
    pylsl.cf_int16: np.int16,
    pylsl.cf_int8: np.int8,
    # pylsl.cf_string:  # For now we don't provide a pre-allocated buffer for string data type.
}


class LSLOutletSettings(ez.Settings):
    stream_name: str | None = None
    stream_type: str | None = None
    map_file: str | None = None  # Path to file containing a list of channel names and locations.


class LSLOutletState(ez.State):
    outlet: pylsl.StreamOutlet | None = None


class LSLOutletUnit(ez.Unit):
    """
    Represents a node in a Labgraph graph that subscribes to messages in a
    Labgraph topic and forwards them by writing to an LSL outlet.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    INPUT_SIGNAL = ez.InputStream(AxisArray)

    SETTINGS: LSLOutletSettings
    STATE: LSLOutletState

    def initialize(self) -> None:
        self._stream_created = False

    def shutdown(self) -> None:
        del self.STATE.outlet
        self.STATE.outlet = None

    @ez.subscriber(INPUT_SIGNAL)
    async def lsl_outlet(self, arr: AxisArray) -> None:
        if self.STATE.outlet is None:
            fs = 1 / arr.axes["time"].gain
            out_shape = [_[0] for _ in zip(arr.shape, arr.dims) if _[1] != "time"]
            out_size = int(np.prod(out_shape))
            info = pylsl.StreamInfo(
                name=self.SETTINGS.stream_name,
                type=self.SETTINGS.stream_type,
                channel_count=out_size,
                nominal_srate=fs,
                channel_format=string2fmt[str(arr.data.dtype)],
                source_id="",  # TODO: Generate a hash from name, type, channel_count, fs, fmt, other metadata...
            )
            # TODO: if self.SETTINGS.map_file: Add channel labels (and locations?) to the info desc.
            self.STATE.outlet = pylsl.StreamOutlet(info)

        if self.STATE.outlet is not None:
            dat = arr.data
            if arr.dims[0] != "time":
                dat = np.moveaxis(dat, arr.dims.index("time"), 0)

            self.STATE.outlet.push_chunk(dat.reshape(dat.shape[0], -1))


class LSLInletSettings(ez.Settings):
    stream_name: str = None
    stream_type: str = None
    local_buffer_dur: float = 1.0
    use_arrival_time: bool = False


class LSLInletState(ez.State):
    resolver: pylsl.ContinuousResolver | None = None
    inlet: pylsl.StreamInlet | None = None
    clock_offset: float = 0.0


class LSLInletUnit(ez.Unit):
    """
    Represents a node in a graph that creates an LSL inlet and
    forwards the pulled data to the unit's output.

    Args:
        stream_name: The `name` of the created LSL outlet.
        stream_type: The `type` of the created LSL outlet.
    """

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    SETTINGS: LSLInletSettings
    STATE: LSLInletState

    def initialize(self) -> None:
        # Build the predicate string. This uses XPATH syntax and can filter on anything in the stream info. e.g.,
        # `"name='BioSemi'" or "type='EEG' and starts-with(name,'BioSemi') and count(info/desc/channel)=32"`
        pred = ""
        if self.SETTINGS.stream_name:
            pred += f"name='{self.SETTINGS.stream_name}'"
        if self.SETTINGS.stream_type:
            if len(pred):
                pred += " and "
            pred += f"type='{self.SETTINGS.stream_type}'"
        self.STATE.resolver = pylsl.ContinuousResolver(pred=pred)
        self._fetch_buffer: np.ndarray | None = None

    def shutdown(self) -> None:
        if self.STATE.inlet is not None:
            self.STATE.inlet.close_stream()
        self.STATE.inlet = None

    def _update_clock_offset(self) -> None:
        pair = (time.time(), pylsl.local_clock())
        new_offset = pair[0] - pair[1]
        self.STATE.clock_offset = new_offset

    @ez.publisher(OUTPUT_SIGNAL)
    async def lsl_pull(self) -> AsyncGenerator:
        while self.STATE.inlet is None:
            results: list[pylsl.StreamInfo] = self.STATE.resolver.results()
            if len(results):
                self.STATE.inlet = pylsl.StreamInlet(
                    results[0],
                    max_chunklen=1,
                    processing_flags=pylsl.proc_ALL
                )
                inlet_info = self.STATE.inlet.info()
                # If possible, create a destination buffer for faster pulls
                fmt = inlet_info.channel_format()
                n_ch = inlet_info.channel_count()
                if fmt in fmt2npdtype:
                    dtype = fmt2npdtype[fmt]
                    n_buff = int(self.SETTINGS.local_buffer_dur * inlet_info.nominal_srate()) or 1000
                    self._fetch_buffer = np.zeros((n_buff, n_ch), dtype=dtype)
                # Pre-allocate a message template.
                fs = inlet_info.nominal_srate()
                self.STATE.msg_template = AxisArray(
                    data=np.empty((0, n_ch)),
                    dims=["time", "ch"],
                    axes={
                        "time": AxisArray.Axis.TimeAxis(fs=fs if fs else 1.0),  # HACK: Use 1.0 for irregular rate.
                    }
                )
                self.STATE.inlet.open_stream()
            else:
                await asyncio.sleep(0.5)

        last_sync_update = time.time() - 1.0
        while self.STATE.inlet is not None:
            if self._fetch_buffer is not None:
                samples, timestamps = self.STATE.inlet.pull_chunk(
                    max_samples=self._fetch_buffer.shape[0],
                    dest_obj=self._fetch_buffer
                )
            else:
                samples, timestamps = self.STATE.inlet.pull_chunk()
                samples = np.array(samples)
            t_now = time.time()
            if (t_now - last_sync_update) >= 1.0:
                self._update_clock_offset()
            if len(timestamps):
                if self.SETTINGS.use_arrival_time:
                    self.STATE.msg_template.axes["time"].offset = t_now
                else:
                    self.STATE.msg_template.axes["time"].offset = timestamps[0] + self.STATE.clock_offset
                view = self._fetch_buffer[:len(timestamps)] if samples is None else samples
                yield self.OUTPUT_SIGNAL, replace(self.STATE.msg_template, data=view)
            else:
                await asyncio.sleep(0.001)
