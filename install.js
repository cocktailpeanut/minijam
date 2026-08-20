const COMFY_COMMIT = "df85c84f7540eafe9e0745ecc163c311204564ad"
const AUDIOCPP_SERVER_SHA256 = "9e4a0447c57a387f4626c9df2a3205c183d9df88fe8ddf1c79bc44a609ac7827"
const AUDIOCPP_MODEL_REVISION = "ead13fe2b1ca3e8ca314d4bdfe1ff04c533f2b13"
const AUDIOCPP_MODEL_FILES = [
  "config/condition_encoder.json",
  "config/language_model.json",
  "config/rvq_depth_decoder.json",
  "config/transformer.json",
  "config/vocoder.json",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
  "condition_encoder.gguf",
  "language_model_q4_0.gguf",
  "rvq_depth_decoder_bf16.gguf",
  "transformer_q4_0.gguf",
  "vocoder.gguf"
]

module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "MiniMax Music 3 requires Apple Silicon on macOS. Intel Macs are not supported."
      }
    },
    {
      when: "{{exists('app/model_cache')}}",
      method: "fs.rm",
      params: {
        path: "app/model_cache"
      }
    },
    {
      when: "{{exists('app/env') && !exists('app/.minimax-only')}}",
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      when: "{{platform !== 'darwin' && !exists('app/comfyui/.git')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: "git clone --no-tags --branch prs/minimax-music-3-graphs --single-branch https://github.com/rattus128/ComfyUI.git comfyui"
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "shell.run",
      params: {
        path: "app/comfyui",
        message: [
          "git -c safe.directory=* fetch origin prs/minimax-music-3-graphs",
          `git -c safe.directory=* checkout --detach ${COMFY_COMMIT}`
        ]
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/comfyui",
        bluefairy: "off",
        message: "uv pip install -r requirements.txt"
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app/comfyui"
        }
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/comfyui",
        message: "uv pip check"
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "hf.download",
      params: {
        path: "app/comfyui/models",
        "_": [
          "Comfy-Org/MiniMax-Music-3",
          "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
          "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
          "vae/minimax_music3_dav.safetensors"
        ],
        "local-dir": "."
      }
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        path: "app",
        message: [
          `echo "${AUDIOCPP_SERVER_SHA256}  audiocpp_server" | shasum -a 256 -c -`,
          "chmod +x audiocpp_server"
        ]
      }
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "hf.download",
      params: {
        path: "app/models",
        "_": ["audio-cpp/MiniMax-Music3-GGUF", ...AUDIOCPP_MODEL_FILES],
        revision: AUDIOCPP_MODEL_REVISION,
        "local-dir": "minimax-music3-gguf"
      }
    },
    {
      when: "{{platform !== 'darwin' || arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "CMAKE_ARGS=\"-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on\" uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
          "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
        ]
      }
    },
    {
      when: "{{platform === 'win32' && arch === 'x64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --upgrade --force-reinstall \"https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19/llama_cpp_python-0.3.19-cp310-cp310-win_amd64.whl\"",
          "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
        ]
      }
    },
    {
      when: "{{platform !== 'darwin' && platform !== 'win32'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20",
          "python -c \"import llama_cpp; print(llama_cpp.__version__)\""
        ]
      }
    },
    {
      when: "{{platform !== 'darwin' || arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python -c \"import gradio, huggingface_hub, requests, soundfile; print('MiniJam app dependencies ready')\""
      }
    },
    {
      when: "{{platform !== 'darwin' || arch === 'arm64'}}",
      method: "fs.write",
      params: {
        path: "app/.minimax-only",
        text: "MiniMax-only runtime\n"
      }
    },
    {
      when: "{{platform !== 'darwin' || arch === 'arm64'}}",
      method: "notify",
      params: {
        html: "MiniJam is installed. Click Start to open it."
      }
    }
  ]
}
