# Excoder

Local copilot based on SantaCoder.

## Usage

### Build

mkdir build
cd build
cmake ..

make excoder excoder-serve

### Run

./excoder-serve -m PATH_TO_SANTACODER -t 8 --temp 0.1

### Client setting

1. Install vscode-fauxpilot plugin.
2. Set as follows:
   "fauxpilot.engine": "santacoder"
   "fauxpilot.server": "http://localhost:18080/v1/engines"
   "fauxpilot.model": "excoder"
   "fauxpilot.temperature": 0.1

## Acknowledgements

Built on ggml and used code from:

https://github.com/bigcode-project/starcoder.cpp

https://github.com/ravenscroftj/turbopilot

Thanks a lot!
