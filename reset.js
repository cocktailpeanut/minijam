module.exports = {
  run: [
    {
      when: "{{exists('app/comfyui')}}",
      method: "fs.rm",
      params: {
        path: "app/comfyui"
      }
    },
    {
      when: "{{exists('app/audiocpp_server')}}",
      method: "fs.rm",
      params: {
        path: "app/audiocpp_server"
      }
    },
    {
      when: "{{exists('app/models')}}",
      method: "fs.rm",
      params: {
        path: "app/models"
      }
    },
    {
      when: "{{exists('app/env')}}",
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      when: "{{exists('app/.minimax-only')}}",
      method: "fs.rm",
      params: {
        path: "app/.minimax-only"
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
      when: "{{exists('app/composer_models')}}",
      method: "fs.rm",
      params: {
        path: "app/composer_models"
      }
    },
    {
      when: "{{exists('app/data')}}",
      method: "fs.rm",
      params: {
        path: "app/data"
      }
    },
    {
      when: "{{exists('cache')}}",
      method: "fs.rm",
      params: {
        path: "cache"
      }
    }
  ]
}
