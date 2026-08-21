module.exports = {
  version: "7.0",
  title: "MiniJam",
  description: "Describe any song in plain English, compose it locally, and generate it with MiniMax Music 3.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env") && (kernel.platform === "darwin"
      ? kernel.arch === "arm64"
        && info.exists("app/audiocpp_server")
        && info.exists("app/models/minimax-music3-gguf/language_model_q4_0.gguf")
        && info.exists("app/models/minimax-music3-gguf/transformer_q4_0.gguf")
      : info.exists("app/comfyui/env")
        && info.exists("app/comfyui/models/diffusion_models/minimax_music3_dit_int8_convrot.safetensors")
        && info.exists("app/comfyui/models/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors")
        && info.exists("app/comfyui/models/vae/minimax_music3_dav.safetensors"))
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open MiniJam",
            href: local.url,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js",
          confirm: "Remove the MiniMax engine, downloaded models, local composer models, and saved songs?"
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
