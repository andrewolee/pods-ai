# Orcasound AI MCP

This server lets you use AI (like Claude) to explore Orcasound data. It helps you find whale calls, check hydrophones, and prepare data for training without writing code.

## Requirements

To run this server, you need:
*   **Python 3.11+**: The core language (matches CI).
*   **Node.js**: Needed to run the "Inspector" or to connect to some AI clients.
*   **uv** (Recommended): For fast and clean setup.
*   **An AI Client**: Like Claude Desktop, Gemini, or VS Code (with an MCP plugin).

> [!NOTE]
> The base `requirements-mcp.txt` installs a minimal set of dependencies for basic data interrogation. Running the model comparison tool (`compare_models_on_clip`) or the spectrogram tool (`generate_spectrogram`) requires the full ML/audio dependencies from the main `requirements.txt`.

## How it works

When you connect this server to an AI (like Gemini or Claude), the AI "reads" the names and descriptions of our tools. 

Because the tool names (like `get_recent_detections`) and their instructions are **fixed** in the code, the AI doesn't guess. It knows exactly what data it's getting and how to use it. This makes the conversation reliable and consistent every time.

---

## Quick Start

<details>
<summary><strong>For Unix</strong></summary>

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Set up the tools**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements-mcp.txt
   ```

   > [!NOTE]
   > If you plan to use `compare_models_on_clip` or `generate_spectrogram`, install the full audio stack from `requirements.txt` instead of relying on the minimal `requirements-mcp.txt` install.

3. **Test it (The "Inspector")**:
   Run this to see a web page with all the tools:
   ```bash
   export PYTHONPATH=src
   export DANGEROUSLY_OMIT_AUTH=true
   npx -y @modelcontextprotocol/inspector .venv/bin/python src/mcp_server.py
   ```

   Open [http://localhost:6274](http://localhost:6274) and click **Connect**.

</details>

<details>
<summary><strong>For Windows</strong></summary>

1. **Install uv**:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Set up the tools**:
   ```powershell
   uv venv .venv
   .venv\Scripts\activate
   uv pip install -r requirements-mcp.txt
   ```

   > [!NOTE]
   > If you plan to use `compare_models_on_clip` or `generate_spectrogram`, install the full audio stack from `requirements.txt` instead of relying on the minimal `requirements-mcp.txt` install.

3. **Test it (The "Inspector")**:
   Run this to see a web page with all the tools:
   ```powershell
   $env:PYTHONPATH="src"
   $env:DANGEROUSLY_OMIT_AUTH="true"
   npx -y @modelcontextprotocol/inspector .venv\Scripts\python.exe src\mcp_server.py
   ```

   Open [http://localhost:6274](http://localhost:6274) and click **Connect**.

</details>

---

## Real Example: Finding new data

Here is exactly how this helps research. I asked an AI ( Gemini-cli): 
> *"Is there any new whale data at Sunset Bay we haven't trained on yet?"*

**The AI did the following automatically:**
1. Looked up the Sunset Bay station.
2. Checked the latest detections from the live website.
3. Compared them to our local training files.
4. Found **31 new whale calls** that were missing!
5. Exported them to `unlabeled_sunset_bay.csv` so I could start using them.

**Example of the generated data:**
```csv
det_032veqonCsn1LcaZ81ifff,2026-04-07T20:09:18.000000Z,human,vessel,Vessel hitting the hydrophone,47340.364191,1775545218
det_032veAu4cz2IRFOO48S6LU,2026-04-07T19:41:44.000000Z,human,other,a train,45685.561536,1775545218
det_032ve9oSUoGKDvj1ClujUY,2026-04-07T19:41:15.000000Z,human,other,A Train,45656.904129,1775545218
det_032uuCAj8ydlSPw76yrYjQ,2026-04-06T12:27:47.000000Z,machine,whale,,19650.0,1775458817
det_032rv284XNvcWjwk6O8VKc,2026-04-01T06:42:30.000000Z,machine,whale,Transient calls,85332.0,1774940418
det_032rcOgki2tf0KtWxR3Vu4,2026-03-31T18:04:00.000000Z,human,whale,,39821.773909,1774940418
```

**Total time:** 15 seconds. (Doing this manually would take more and we would need to run more than one different scripts and SQL queries).

---

## Using it with Claude Desktop

To use these tools inside Claude, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orcasound": {
      "command": "/absolute/path/to/pods-ai/.venv/bin/python",
      "args": ["/absolute/path/to/pods-ai/src/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/pods-ai/src"
      }
    }
  }
}
```

Now you can just ask Claude: *"Find me new whale calls at Orcasound Lab and save them to a CSV."*

---

## Using it with Visual Studio

Visual Studio 2022 supports MCP servers through GitHub Copilot Chat.

1. **Configure GitHub Copilot Chat** to use the MCP server:

Create or edit `.github/copilot-instructions.md` in your repository to include MCP server information, or configure it through Visual Studio settings if MCP configuration UI is available.

Alternatively, if your Visual Studio version supports MCP configuration files, create `.vscode/settings.json` (for VS Code compatibility) or wait for native Visual Studio MCP configuration support.

2. Open **Copilot Chat** (View > GitHub Copilot Chat, or `Ctrl+Alt+/`).

3. Now you can just ask Copilot: *"Find me new whale calls at Orcasound Lab and save them to a CSV."*

### Troubleshooting

- **Connection Issues**: Check that `requirements-mcp.txt` dependencies are installed.
- **Path Problems**: Ensure `PYTHONPATH` includes the `src` directory.
- **Server Not Responding**: Verify the virtual environment is activated and Python version is 3.11+.
- **Logs**: Check Visual Studio's Output window for MCP server logs.

### Current Limitations

> [!NOTE]
> As of 2026, native Visual Studio MCP support is still evolving. The dual-handshake implementation ensures forward compatibility. If you encounter issues, use the MCP Inspector or Claude Desktop for testing while waiting for full Visual Studio integration.

---

## Available Tools

The MCP server exposes these tools to AI assistants:

1. **`list_hydrophones`** - Get all active Orcasound hydrophone stations
2. **`get_recent_detections`** - Fetch latest detections from a specific station
3. **`list_s3_recordings`** - Browse available audio recordings in S3
4. **`get_sample_stats`** - Check training/testing data distribution
5. **`find_unlabeled_detections`** - Find new detections not in training data
6. **`compare_models_on_clip`** - Test OrcaHello and PODS-AI models on audio
7. **`generate_spectrogram`** - Generate a PNG spectrogram for a local WAV clip
8. **`export_unlabeled_to_csv`** - Save new detections directly to CSV

All tools work without AWS credentials (public data). `compare_models_on_clip` and `generate_spectrogram` require the full ML/audio dependencies from `requirements.txt`.
