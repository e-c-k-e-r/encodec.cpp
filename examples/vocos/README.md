# vocos.cpp

High-performance inference of [Vocos]() vocoder using Encodec codes:

- Plain C/C++ implementation without dependencies using [ggml](https://github.com/ggerganov/ggml)

Currently errors out at `vocos.cpp#L641`.

# Download Vocos weights from HuggingFace Hub

```python

```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
