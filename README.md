# MCP Securities Analysis

A Python-based flow for securities analysis using the **Model Context Protocol (MCP)**.  The repository bundles market data, parsing, analytics and visualisation tools behind a FastMCP server so that they can be consumed locally or remotely by any MCP-aware client (e.g. Claude Desktop, LangChain, OpenAI-Function calling, etc.).

[Example deep research report for Tesla](https://claude.ai/public/artifacts/2f1df8b6-ffbc-40ca-a2d9-6d068bdb01a9).

This was generated semi-autonomously by the following steps:

- connect MCP tools to Claude Desktop, including web search, Perplexity, Wikipedia, in addition to the market data tools in server.py for fundamental, technical analysis, and news search.

- prompt Claude Desktop to query Perplexity, Wikipedia, and the 10-K to write a profile of Tesla

- prompt Claude Desktop to query each tool for info on Tesla

- finally, enable deep research and prompt Claude Desktop to write a deep report in 8 sections with details on what each section should cover, using the information retrieved from the tools.

While it's not a fully autonomous agent and at an early POC level, it shows clear path toward a fully autonomous agent. Create an MCP client that goes through the steps above and generates a deep report on Tesla in a structured format with graphs and tables. And then create an even more advanced [multi-agent workflow](https://www.anthropic.com/engineering/built-multi-agent-research-system) with a set of parallel agents for each section, and a critic-optimizer workflow, and a final report generator.

---

## Features

- **FastMCP server** – exposes a few MCP *tools* to get market data, news, charts, SEC filings, fundamental, technical data, research from public web sites, subscription services, and REST APIs.

- **Market data** – real-time and historical OHLCV data via `yfinance` & `OpenBB`.

- **Fundamental data** – automatic downloading of SEC filings (`sec_downloader`) and rich XBRL/HTML parsing through `sec_parser`.

- **News & Social sentiment** – headlines with `newsapi-python` plus Reddit scraping utilities.

- **Technical analysis** – hundreds of indicators with `pandas_ta` & `TA-Lib`.

- **Interactive plots** – high-quality Plotly charts exported server-side (static PNG or interactive HTML).

- **Async-first design** – built on `asyncio`, `aiohttp`, `httpx` & Playwright for maximum throughput.

This section AI-generated so beware of hype. New project, would like to share and get comments, not extensively tested. Use it as a starting point, at your own risk.

---

## Quick Start

```bash
# 1. Clone and enter the project
$ git clone https://github.com/<your-org>/MCP.git
$ cd MCP

# 2. Create & activate a virtualenv (recommended)
$ python -m venv .venv
$ source .venv/bin/activate

# 3. Install python dependencies
$ pip install -r requirements.txt

# 4. Install Playwright browsers (once)
$ playwright install

# 5. Copy environment template & add your keys
$ cp dotenv.txt .env  # then edit as needed

# 6. Launch and test the server
$ LOGLEVEL=DEBUG mcp dev server.py
# click to the link in the terminal to open the test page, connect, view tools, and then test them individually

# 7. Use the server in your MCP client of choice. For Claude Desktop, edit the provided claude_desktop_config.json file and move it to the proper location for your platform (macOS, Windows).
https://claude.ai/download
https://modelcontextprotocol.io/quickstart/user

# 8. For additional optional market data MCP servers install these repos
Wikipedia MCP (no API key required)
pip install wikipedia-mcp

Filesystem MCP (use local files, no API key required)
git clone https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

Yahoo Finance MCP (no API key required)
git clone https://github.com/Alex2Yang97/yahoo-finance-mcp.git

FMP MCP (API key required)
git clone https://github.com/cdtait/fmp-mcp-server

Alpha Vantage MCP (API key required)
git clone https://github.com/calvernaz/alphavantage.git
# see alphavantage.patches for edits to alphavantage/src/alphavantage_mcp_server/server.py

Brave Search MCP (API key required)
npm install -g @modelcontextprotocol/server-brave-search

Perplexity Ask MCP (API key, Docker required)
Follow setup instructions here:
https://github.com/ppl-ai/modelcontextprotocol/tree/main

```

- [MCP / Claude Desktop Quickstart](https://modelcontextprotocol.io/quickstart/user)

- [Awesome MCP Servers](https://awesome-mcp-servers.com/)

---

## Project Structure

```bash
MCP/
├── claude_desktop_config.json # Configuration for Claude Desktop
├── dotenv.txt             # Secrets / Environment variables
├── README.md              # This file
├── server.py              # FastMCP server, launched by mcp dev or Claude desktop or other MCP client
├── requirements.txt       # Python dependencies
├── sources.yaml           # Data-source configuration used by server
├── Market Data.ipynb      # Jupyter notebook to fetch market data
├── TearSheet.ipynb        # Jupyter notebook to do basic analysis

```
