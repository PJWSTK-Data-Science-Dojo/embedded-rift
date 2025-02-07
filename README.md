# Embedded Rift

## Project Setup

### Prerequisites

To run this project, you need the following API keys:
- **OpenAI API Key**: Obtain an API key by signing up at [OpenAI](https://platform.openai.com/).
- **Riot API Key**: Obtain an API key by signing up at [Riot Games Developer Portal](https://developer.riotgames.com/).
- **Docker & Docker-Compose**: 
  - **Docker**: Required to run MongoDB in a containerized environment. Download and install it from [Docker Desktop](https://www.docker.com/products/docker-desktop/).
  - **Docker-Compose**: A tool for defining and running multi-container Docker applications. It usually comes with Docker but if needed, install it separately by following [Docker-Compose](https://docs.docker.com/compose/install/) Installation.
- **Mongo Tools**: [Mongo Tools Download .msi](https://fastdl.mongodb.org/tools/db/mongodb-database-tools-windows-x86_64-100.11.0.msi)

Once you have the keys, add them to the `.env` file as follows:
```env
OPENAI_API_KEY=your_openai_api_key_here
RIOT_API_KEY=your_riot_api_key_here
MONGO_URI=
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

### 7. Setup MongoDB with Docker
The project includes a `docker-compose.yml` file that sets up **MongoDB.**

## 7.1 Start MongoDB
Run the following command in the project root:
```bash
docker-compose up -d
```

## 7.2 Import Database Backup
You need to download the backup file (`games_mongo.gz`) and import it into MongoDB.

1. Copy `games_mongo.gz` into the project root directory.
2. Run the following command to restore the database:
```bash
mongorestore --gzip --archive=/games_mongo.gz --nsInclude=embedded-rift.games
```

