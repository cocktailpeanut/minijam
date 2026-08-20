module.exports = {
  daemon: true,
  run: [
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        path: "app",
        env: {
          MM3_SOLVER: "ab2"
        },
        message: "./audiocpp_server --config {{os.totalmem() >= 32000000000 ? 'audio.cpp-server-fast.json' : 'audio.cpp-server.json'}} --host 127.0.0.1 --port {{port}} --backend metal",
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "local.set",
      params: {
        engine_url: "{{input.event[1]}}",
        engine_type: "audiocpp"
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/comfyui",
        env: {
          TOKENIZERS_PARALLELISM: "false"
        },
        message: "{{platform === 'win32' && gpu === 'amd' ? 'python main.py --directml' : 'python main.py'}} --listen 127.0.0.1 --port {{port}} --disable-api-nodes",
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      when: "{{platform !== 'darwin'}}",
      method: "local.set",
      params: {
        engine_url: "{{input.event[1]}}",
        engine_type: "comfyui"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          GRADIO_ANALYTICS_ENABLED: "False",
          GRADIO_SERVER_NAME: "127.0.0.1",
          PYTHONUNBUFFERED: "1",
          MINIMAX_ENGINE_URL: "{{local.engine_url}}",
          MINIMAX_ENGINE: "{{local.engine_type}}",
          MINIMAX_AUDIOCPP_MEMORY_MODE: "{{platform === 'darwin' && os.totalmem() >= 32000000000 ? 'fast' : platform === 'darwin' ? 'low-memory' : 'dynamic-vram'}}",
          XDG_CACHE_HOME: "{{path.resolve(cwd, 'cache')}}",
          HF_HOME: "{{path.resolve(cwd, 'cache', 'huggingface')}}",
          HF_MODULES_CACHE: "{{path.resolve(cwd, 'cache', 'hf_modules')}}",
          MPLCONFIGDIR: "{{path.resolve(cwd, 'cache', 'matplotlib')}}",
          LLAMA_CACHE: "{{path.resolve(cwd, 'cache', 'llama')}}"
        },
        message: "python app.py",
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
