module.exports = {
  run: [
    {
      when: "{{gpu === 'nvidia' && platform === 'win32' && Number.parseFloat(gpu_driver || '0') >= 580 && Number.parseInt((gpu_target || '').replace('sm_', ''), 10) >= 75 && !/gtx\\s*1[06]/i.test(gpu_model || '')}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        bluefairy: "off",
        message: "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps"
      },
      next: null
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'linux' && Number.parseFloat(gpu_driver || '0') >= 580 && Number.parseInt((gpu_target || '').replace('sm_', ''), 10) >= 75 && !/gtx\\s*1[06]/i.test(gpu_model || '')}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        bluefairy: "off",
        message: "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall"
      },
      next: null
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        bluefairy: "off",
        message: "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps"
      },
      next: null
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        bluefairy: "off",
        message: "uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall"
      },
      next: null
    },
    {
      when: "{{gpu === 'amd' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torch-directml torchaudio torchvision numpy==1.26.4 --force-reinstall"
      },
      next: null
    },
    {
      when: "{{gpu === 'amd' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4 --force-reinstall --no-deps"
      },
      next: null
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --force-reinstall --no-deps"
      },
      next: null
    },
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps",
          "uv pip install numpy==1.26.4 --force-reinstall --no-deps"
        ]
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
      }
    }
  ]
}
