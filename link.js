module.exports = {
  run: [
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    },
    {
      when: "{{exists('app/comfyui/env')}}",
      method: "fs.link",
      params: {
        venv: "app/comfyui/env"
      }
    }
  ]
}
