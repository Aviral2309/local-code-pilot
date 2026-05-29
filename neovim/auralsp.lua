-- auralsp.lua
-- Neovim LSP configuration for AuraLSP
--
-- Installation:
--   1. Copy this file to ~/.config/nvim/lua/auralsp.lua
--   2. Add `require('auralsp')` to your init.lua
--   3. Ensure `auralsp` command is on your PATH (pip install -e .)
--
-- The LSP client communicates via stdin/stdout JSON-RPC.
-- Neovim's built-in lspconfig handles the lifecycle automatically.

local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Register AuraLSP as a custom LSP server
if not configs.auralsp then
  configs.auralsp = {
    default_config = {
      -- Command that starts the LSP server process
      -- Neovim spawns this and communicates via stdin/stdout
      cmd = { 'python', '-m', 'src.server' },

      -- Which file types trigger the LSP
      filetypes = { 'python', 'javascript', 'typescript', 'go', 'rust' },

      -- Root directory detection: look for these marker files
      root_dir = lspconfig.util.root_pattern(
        'pyproject.toml',
        'setup.py',
        'package.json',
        'Cargo.toml',
        'go.mod',
        '.git'
      ),

      settings = {},

      init_options = {
        -- Pass the config path if you want a non-default location
        -- config_path = vim.fn.expand('~/.config/auralsp/config.json'),
      },
    },
  }
end

-- Setup with nvim-cmp completion capabilities
local capabilities = vim.lsp.protocol.make_client_capabilities()
capabilities.textDocument.completion.completionItem.snippetSupport = true

lspconfig.auralsp.setup({
  capabilities = capabilities,
  on_attach = function(client, bufnr)
    vim.notify('AuraLSP attached to ' .. vim.api.nvim_buf_get_name(bufnr), vim.log.levels.INFO)

    -- Keymap: trigger completion manually with <C-Space>
    vim.keymap.set('i', '<C-Space>', function()
      vim.lsp.buf.completion()
    end, { buffer = bufnr, desc = 'AuraLSP: trigger completion' })
  end,
  on_exit = function(code, signal, client_id)
    if code ~= 0 then
      vim.notify('AuraLSP exited with code ' .. code, vim.log.levels.WARN)
    end
  end,
})
