from typing import Optional

import ezmsg.core as ez
from ezmsg.lsl.units import LSLInletUnit, LSLInletSettings
from ezmsg.util.debuglog import DebugLog, DebugLogSettings
# from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings


class LSLDemoSystemSettings(ez.Settings):
    stream_name: str = ""
    stream_type: str = "EEG"
    logger_name: str = "DEBUG"  # Useful name for logger
    logger_max_length: Optional[int] = 400  # No limit if `None``


class LSLDemoSystem(ez.Collection):
    SETTINGS: LSLDemoSystemSettings

    INLET = LSLInletUnit()
    LOGGER = DebugLog()
    # LOGGER = MessageLogger()

    def configure(self) -> None:
        self.INLET.apply_settings(
            LSLInletSettings(
                stream_name=self.SETTINGS.stream_name,
                stream_type=self.SETTINGS.stream_type
            )
        )
        self.LOGGER.apply_settings(
            DebugLogSettings(
                name=self.SETTINGS.logger_name,
                max_length=self.SETTINGS.logger_max_length
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INLET.OUTPUT_SIGNAL, self.LOGGER.INPUT),
        )


if __name__ == "__main__":
    # Run the websocket system
    system = LSLDemoSystem()
    system.apply_settings(
        LSLDemoSystemSettings(stream_name="", stream_type="EEG")
    )
    ez.run(SYSTEM=system)
