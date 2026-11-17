"""Camera configuration, loaded from .env. Never log the real RTSP URL."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Tapo cameras expose a high-resolution and a low-resolution stream.
STREAMS = {"hi": "stream1", "lo": "stream2"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class CameraConfig:
    host: str
    port: int
    user: str
    password: str

    def rtsp_url(self, stream: str = "hi") -> str:
        """Real URL with credentials. Pass to the decoder only; never print."""
        user = quote(self.user, safe="")
        password = quote(self.password, safe="")
        return f"rtsp://{user}:{password}@{self.host}:{self.port}/{STREAMS[stream]}"

    def masked_url(self, stream: str = "hi") -> str:
        """Safe to print."""
        return f"rtsp://***:***@{self.host}:{self.port}/{STREAMS[stream]}"

    def redact(self, text: str) -> str:
        """Strip credentials from arbitrary text (e.g. decoder error messages)."""
        for secret in (self.password, quote(self.password, safe="")):
            if secret:
                text = text.replace(secret, "***")
        return text


def load_camera_config() -> CameraConfig:
    env_file = ROOT / ".env"
    if not env_file.exists():
        raise ConfigError(
            f"No .env file at {env_file}. Copy .env.example to .env and fill in "
            "the Tapo camera account."
        )
    load_dotenv(env_file, override=True)

    missing = [k for k in ("TAPO_HOST", "TAPO_USER", "TAPO_PASSWORD") if not os.getenv(k)]
    if missing:
        raise ConfigError(f"Missing in .env: {', '.join(missing)}")

    port = os.getenv("TAPO_PORT") or "554"
    if not port.isdigit():
        raise ConfigError(f"TAPO_PORT must be a number, got {port!r}")

    return CameraConfig(
        host=os.environ["TAPO_HOST"],
        port=int(port),
        user=os.environ["TAPO_USER"],
        password=os.environ["TAPO_PASSWORD"],
    )
