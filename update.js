module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "script.start",
      params: {
        uri: "install.js"
      }
    },
    {
      method: "notify",
      params: {
        html: "MiniJam is up to date."
      }
    }
  ]
}
