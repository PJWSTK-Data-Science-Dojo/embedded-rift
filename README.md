# Embedded Rift

## Project Setup

### Prerequisites

To run this project, you need the following API keys:
- **OpenAI API Key**: Obtain an API key by signing up at [OpenAI](https://platform.openai.com/).
- **Riot API Key**: Obtain an API key by signing up at [Riot Games Developer Portal](https://developer.riotgames.com/).

Once you have the keys, add them to the `.env` file as follows:
```env
OPENAI_API_KEY=your_openai_api_key_here
RIOT_API_KEY=your_riot_api_key_here
```

### 1. Create a Virtual Environment
```bash
# Windows command line
python -m venv .venv
.venv\Scripts\activate

# Windows Git Bash
python -m venv .venv
source .venv/Scripts/activate

# MacOS, Linux and Git Bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Enviroment Variables
```bash
cp .env.example .env
```

### 4. Set the Python Package Path
You can temporarily set the `PYTHONPATH` to include the `src` directory. This ensures that Python knows where to find your modules.
*For Windows:*
```bash
set PYTHONPATH=src
```
*For MacOS, Linux and GitBash:*
```bash
export PYTHONPATH=src
```
***Optional (Not Recommended): Add PYTHONPATH Permanently***

You can permanently add the src directory to your Python path in your shell configuration file:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/src
```

### 5. Add Data and Visualization Catalogs
```bash
mkdir data
mkdir viz
```

### 6. Run the Script
Test the setup by running one of the scripts:
```bash
python src/scripts/test.py
```