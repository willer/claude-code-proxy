# Anthropic API Proxy for LiteLLM Models 🔄

**Use Anthropic clients (like Claude Code) with any LiteLLM-supported model.** 🤝

A proxy server that lets you use Anthropic clients with OpenAI, Gemini, or other models supported by LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Google AI Studio (Gemini) API key (if using Google provider) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `OPENAI_API_KEY`: Your OpenAI API key (Required for OpenAI models).
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required for Gemini models).
   *   `BIG_MODEL` (Optional): The full model identifier (including provider prefix) to map `sonnet` requests to. Defaults to `openai/gpt-4.1`.
   *   `SMALL_MODEL` (Optional): The full model identifier (including provider prefix) to map `haiku` requests to. Defaults to `openai/gpt-4.1-mini`.

   **Mapping Logic:**
   - `haiku` requests map to the model specified in `SMALL_MODEL`
   - `sonnet` requests map to the model specified in `BIG_MODEL`
   - Models without a provider prefix are automatically prefixed with `openai/`
   - Models with existing provider prefixes (`openai/`, `gemini/`, `anthropic/`) are used as-is

4. **Run the server**:
   ```bash
   ./run-server.sh
   ```
   *(the script sets sensible defaults, including `--reload`, and can be customised by editing environment variables inside)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```
   
   Or use the included `claudeo` script:
   ```bash
   # Add execute permissions
   chmod +x claudeo
   
   # Run Claude with the proxy
   ./claudeo
   ```

3. **That's it!** Your Claude Code client will now use the configured LiteLLM-supported models through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude model aliases and handles provider prefixes:

| Claude Model | Maps To |
|--------------|---------|
| haiku | `SMALL_MODEL` environment variable (default: openai/gpt-4.1-mini) |
| sonnet | `BIG_MODEL` environment variable (default: openai/gpt-4.1) |
| thinker (invoked by thinking budget) | `THINKER_MODEL` environment variable (default: openai/gpt-4o) |

**Special Values:**
- Setting `BIG_MODEL="passthrough"` enables a special mode where all Claude/Anthropic models bypass LiteLLM and go directly to the Anthropic API. This preserves all original request parameters and ensures maximum compatibility.

### Model Prefix Handling

Models can be specified with explicit provider prefixes to control which provider they use:

- `openai/gpt-4o` - Uses OpenAI's GPT-4o model
- `gemini/gemini-2.5-pro` - Uses Gemini's 2.5 Pro model
- `anthropic/claude-3-opus` - Uses Anthropic's Claude 3 Opus model

Models without a prefix automatically get the `openai/` prefix added:

- `gpt-4o` becomes `openai/gpt-4o`
- `gpt-4.1-mini` becomes `openai/gpt-4.1-mini`

The API selects the appropriate backend and API key based on the provider prefix in the model name.

### Customizing Model Mapping

Control the mapping using environment variables in your `.env` file or directly:

**Example 1: Default Setup**
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key" # Needed only for gemini/ models
ANTHROPIC_API_KEY="your-anthropic-key" # Needed only for anthropic/ models
# BIG_MODEL="openai/gpt-4.1" # Optional, it's the default
# SMALL_MODEL="openai/gpt-4.1-mini" # Optional, it's the default
```

**Example 2: Use Gemini Models for Claude Aliases**
```dotenv
OPENAI_API_KEY="your-openai-key" # Needed for openai/ models
GEMINI_API_KEY="your-google-key" # Needed for gemini/ models
BIG_MODEL="gemini/gemini-2.5-pro-preview-03-25" # Maps 'sonnet' to Gemini
SMALL_MODEL="gemini/gemini-2.0-flash" # Maps 'haiku' to Gemini
```

**Example 3: Use Specific OpenAI Models**
```dotenv
OPENAI_API_KEY="your-openai-key"
BIG_MODEL="openai/gpt-4o" # Maps 'sonnet' to GPT-4o
SMALL_MODEL="openai/gpt-4o-mini" # Maps 'haiku' to GPT-4o-mini
```

**Example 4: Passthrough Mode for Direct Anthropic API Access**
```dotenv
ANTHROPIC_API_KEY="your-anthropic-key" # Required for passthrough mode
OPENAI_API_KEY="your-openai-key" # Needed for non-Anthropic models
BIG_MODEL="passthrough" # Special value that bypasses LiteLLM for Anthropic models
SMALL_MODEL="openai/gpt-4o-mini" # Maps 'haiku' to GPT-4o-mini
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Processing model names** to handle provider prefixes and Claude aliases 🔄

**For normal mode (using LiteLLM):**
3. **Translating** the requests to LiteLLM format (with appropriate provider prefixes) 🔄 
4. **Selecting** the appropriate API key based on the provider prefix 🔑
5. **Sending** the translated request to the selected backend via LiteLLM 📤
6. **Converting** the response back to Anthropic format 🔄

**For passthrough mode (when BIG_MODEL="passthrough"):**
3. **Detecting** Anthropic model requests and bypassing LiteLLM 🚦
4. **Cleaning** the request to ensure API compatibility 🧹
5. **Forwarding** the request directly to the Anthropic API 📤
6. **Returning** the native Anthropic response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients while providing access to any model supported by LiteLLM or direct Anthropic API access. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
