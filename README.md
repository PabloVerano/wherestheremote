# wherestheremote

Locate the Apple TV remote in the living room using a Tapo C225 camera and,
later, a Bluetooth motion tag on the remote. Runs locally; no video is stored.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the Tapo camera account
```

`.env` holds the camera password and is git-ignored.

## Commands

| Command | What it does |
|---|---|
| `uv run python -m wtr.check_stream` | Connects to both camera streams, prints codec / resolution / fps, saves a frame to `snapshots/` |
| `uv run python -m wtr.benchmark` | Times the candidate models on this machine's CPU (`--backend openvino` on Intel) |

Benchmark numbers only mean something on the machine that will run the
tracker (the Umbrel PC), not on the development Mac.

## Status

- [x] Repo scaffold, camera config, stream check, model benchmark
- [ ] Stream check run against the real camera
- [ ] Benchmark run on the Umbrel PC
- [ ] Remote detection + wrist tracking + room-photo screen
- [ ] BLE motion tag fusion
- [ ] Docker packaging for Umbrel
