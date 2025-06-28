"""MCP (Model Context Protocol) Server Implementation.

This module implements a simple MCP server using FastMCP, providing:
- Basic arithmetic operations (e.g., addition)
- Dynamic resource generation (e.g., personalized greetings)

The server follows the Model Context Protocol specification as defined at:
https://modelcontextprotocol.io/

Example:
    To start the server:
    ```
    python server.py
    ```

    Then interact with the server using the MCP client or via HTTP requests:
    - Tool endpoint: POST http://localhost:8000/tools/add
    - Resource endpoint: GET http://localhost:8000/resources/greeting/World

Install into claude_desktop_config.json with:
$ mcp install server.py

Run for debugging with:
$ LOGLEVEL=DEBUG mcp dev server.py

Run for production with (or just use in Claude Desktop):
$ mcp run server.py
"""

# test fetch url , get morningstar with login
import secrets
import time
import re
import random
import os
import logging
import sys

from typing_extensions import Annotated
from trafilatura import extract
import sec_parser as sp
from sec_downloader import Downloader
from playwright.async_api import (
    async_playwright,
    BrowserContext as AsyncBrowserContext,
    Page as AsyncPage
)
from plotly.subplots import make_subplots
from PIL import Image as PILImage
from bs4 import BeautifulSoup
import yfinance as yf
from mcp.server.fastmcp import FastMCP, Image as MCPImage
import yaml
import io
import plotly.io as pio
import plotly.graph_objects as go
import httpx
import dotenv
from typing import Any, Dict, List, Optional, AsyncIterator
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from openbb import obb
import pandas as pd
import pandas_ta as ta
import aiohttp
import importlib

# import openbb
# from openbb_core.app.model.obbject import OBBject

# import base64


# from mcp.types import Resource
# from mcp.server import Server

# from pydantic import Field  # BaseModel, DirectoryPath

# Load environment variables
dotenv.load_dotenv()


FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/j6cl7lzz.playwright'
SOURCES = "sources.yaml"

# Change working directory to the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

logger = logging.getLogger(__name__)
logger.info("Starting in directory: %s", os.getcwd())


async def get_browser(playwright):
    """
    Initializes a Playwright browser instance with stealth settings.

    Args:
        playwright: The Playwright instance.

    Returns:
        Browser: The initialized browser instance.
    """
    viewport = random.choice([
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720}
    ])

    # random device-scale-factor for additional randomization
    device_scale_factor = random.choice([1, 1.25, 1.5, 1.75, 2])

    # Random color scheme and timezone
    color_scheme = random.choice(['light', 'dark', 'no-preference'])
    timezone_id = random.choice([
        'America/New_York', 'Europe/London', 'Europe/Paris',
        'Asia/Tokyo', 'Australia/Sydney', 'America/Los_Angeles'
    ])
    locale = random.choice([
        'en-US', 'en-GB'
    ])
    extra_http_headers = {
        "Accept-Language": f"{locale.split('-')[0]},{locale};q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1" if random.choice([True, False]) else "0"
    }

    return await playwright.firefox.launch_persistent_context(
        user_data_dir=FIREFOX_PROFILE_PATH,
        headless=False,  # run headless, hide splash window
        viewport=viewport,
        device_scale_factor=device_scale_factor,
        timezone_id=timezone_id,
        color_scheme=color_scheme,
        extra_http_headers=extra_http_headers,
        # removes Playwright's default flag
        ignore_default_args=["--enable-automation"],
        args=[
            "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"
        ],
        # provide a valid realistic User-Agent string for the latest Firefox on Apple Silicon
        user_agent="Mozilla/5.0 (Macintosh; ARM Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
        accept_downloads=True,
    )


def perform_human_like_actions(page):
    """Perform random human-like actions on a page to mimic real user behavior."""
    # Random mouse movements
    for _ in range(random.randint(3, 8)):
        # Move mouse with multiple steps to simulate human-like movement
        x = random.randint(100, 1200)
        y = random.randint(100, 700)
        steps = random.randint(5, 10)

        # Calculate increments for smooth movement
        for step in range(1, steps + 1):
            next_x = x * step / steps
            next_y = y * step / steps

            # Add slight randomness to path
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-5, 5)

            page.mouse.move(next_x + jitter_x, next_y + jitter_y)
            time.sleep(random.uniform(0.01, 0.05))

    # Random scrolling behavior
    scroll_amount = random.randint(300, 700)
    page.evaluate(f"window.scrollBy(0, {scroll_amount})")
    time.sleep(random.uniform(0.5, 2))

    # Sometimes scroll back up a bit
    if random.random() > 0.7:
        page.evaluate(f"window.scrollBy(0, -{random.randint(100, 300)})")
        time.sleep(random.uniform(0.3, 1))

    return page


def normalize_html(path: Path | str) -> str:
    """
    Clean and extract text content from an HTML file, including titles and social media metadata.

    Args:
        path (Path | str): Path to the HTML file to process

    Returns:
        - str: Extracted and cleaned text content, or empty string if processing fails

    The function extracts:
        - Page title from <title> tag
        - Social media titles from OpenGraph and Twitter meta tags
        - Social media descriptions from OpenGraph and Twitter meta tags
        - Main content using trafilatura library

    All extracted content is concatenated and truncated to MAX_INPUT_TOKENS length.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as exc:
        print(f"Error: {str(exc)}")
        print(f"Skipping {path}")
        return ""

    # Parse the HTML content using trafilatura
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        # Try to get the title from the <title> tag
        title_tag = soup.find("title")
        title_str = "Page title: " + title_tag.string.strip() + \
            "\n" if title_tag and title_tag.string else ""
    except Exception as exc:
        title_str = ""
        print(str(exc), "clean_html page_title")

    try:
        # Try to get the title from the Open Graph meta tag
        og_title_tag = soup.find("meta", property="og:title")
        if not og_title_tag:
            og_title_tag = soup.find(
                "meta", attrs={"name": "twitter:title"})
        og_title = og_title_tag["content"].strip(
        ) + "\n" if og_title_tag and og_title_tag.get("content") else ""
        og_title = "Social card title: " + og_title if og_title else ""
    except Exception as exc:
        og_title = ""
        print(str(exc), "clean_html og_title")

    try:
        # get summary from social media cards
        og_desc_tag = soup.find("meta", property="og:description")
        if not og_desc_tag:
            # Extract the Twitter description
            og_desc_tag = soup.find(
                "meta", attrs={"name": "twitter:description"})
        og_desc = og_desc_tag.get("content").strip() + \
            "\n" if og_desc_tag else ""
        og_desc = 'Social card description: ' + og_desc if og_desc else ""
    except Exception as exc:
        og_desc = ""
        print(str(exc), "clean_html og_desc")

    # Get text and strip leading/trailing whitespace
    print(title_str + og_title + og_desc, "clean_html")
    try:
        plaintext = extract(html_content)
        plaintext = plaintext.strip() if plaintext else ""
    except Exception as exc:
        plaintext = html_content
        print(str(exc), "clean_html trafilatura")

    # remove special tokens, have found in artiles about tokenization
    # All OpenAI special tokens follow the pattern <|something|>
    special_token_re = re.compile(r"<\|\w+\|>")
    plaintext = special_token_re.sub("", plaintext)
    visible_text = title_str + og_title + og_desc + plaintext
    return visible_text


@dataclass
class SourceConfig:
    """Configuration for a data source used in web scraping.

    Attributes:
        name (str): The name of the data source.
        description (str): A description of the data source, can be used for a tool description.
        url_template (str): Template string for constructing the source URL.
        exchange_mappings (str, optional): Mapping of exchange names to their corresponding values in the URL template.
        required_params (List[str]): List of required parameters for the URL template.
        wait_strategy (str, optional): Page load strategy (e.g., 'load', 'domcontentloaded').
            Defaults to "load".
        rate_limit (float, optional): Minimum delay between requests in seconds.
            Defaults to 2.0.
        priority (int, optional): Priority of the source (lower number = higher priority).
            Defaults to 2.
        data_type (str, optional): Type of data provided by the source.
            Defaults to "general".
    """
    name: str
    description: str
    url_template: str
    required_params: List[str]
    exchange_mappings: Optional[str] = None
    wait_strategy: str = "load"
    rate_limit: float = 2.0
    data_type: str = "general"


@dataclass
class BrowserContext:
    """Browser context for web scraping."""
    browser: AsyncBrowserContext
    page_pool: List[AsyncPage] = field(default_factory=list)
    last_request_time: Dict[str, float] = field(default_factory=dict)
    sources: Dict[str, SourceConfig] = field(default_factory=dict)
    exchanges: Dict[str, Dict[str, str]] = field(default_factory=dict)


class StockDataExtractor:
    """Stock data extractor for web scraping."""

    def __init__(self, context: BrowserContext):
        self.context = context

    async def get_or_create_page(self) -> AsyncPage:
        """Get a page from the pool or create a new one."""
        if self.context.page_pool:
            return self.context.page_pool.pop()
        if not hasattr(self.context, 'browser') or self.context.browser is None:
            raise RuntimeError(
                "Browser is not initialized. Call browser_lifespan() first.")
        return await self.context.browser.new_page()

    async def return_page(self, page: AsyncPage):
        """Return a page to the pool for reuse."""
        try:
            # Clear the page state
            await page.goto("about:blank")
            await page.evaluate(
                "() => { localStorage.clear(); sessionStorage.clear(); }")
            self.context.page_pool.append(page)
        except Exception as e:
            print(f"Error returning page to pool: {str(e)}")
            # If cleaning fails, close the page
            await page.close()

    def respect_rate_limit(self, source_key: str, rate_limit: float):
        """Ensure we respect rate limits for each source."""
        now = time.time()
        last_request = self.context.last_request_time.get(source_key, 0)
        time_since_last = now - last_request

        if time_since_last < rate_limit:
            sleep_time = rate_limit - time_since_last
            time.sleep(sleep_time)

        self.context.last_request_time[source_key] = time.time()

    def map_exchange(self, source_key: str, exchange: str) -> str:
        """Map exchange codes for specific platforms."""
        mappings = self.context.exchanges.get(source_key, {})
        return mappings.get(exchange, exchange)


def load_sources_config(config_path: str = SOURCES) -> tuple[Dict[str, SourceConfig], Dict[str, Dict[str, str]]]:
    """Load source configurations from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sources = {}
    for key, source_data in config['sources'].items():
        sources[key] = SourceConfig(**source_data)

    exchange_mappings = config.get('exchange_mappings', {})

    return sources, exchange_mappings


@asynccontextmanager
async def browser_lifespan(_server: FastMCP) -> AsyncIterator[BrowserContext]:
    """Manage browser lifecycle with configuration loading.

    Args:
        _server: The MCP server instance (unused in this implementation)

    Yields:
        BrowserContext: The initialized browser context
    """
    try:
        # Load configuration
        logger.info("Loading configuration...")
        loaded_sources, loaded_exchanges = load_sources_config()
        logger.info("Successfully loaded configuration")

        # Log loaded exchanges
        for exchange_name in loaded_exchanges:
            logger.info("Loaded exchange mapping for %s", exchange_name)

        # Initialize browser
        logger.info("Initializing Playwright browser...")
        async with async_playwright() as playwright:
            browser = None
            try:
                logger.info("Launching browser...")
                browser = await get_browser(playwright)
                logger.info("Browser launched successfully")

                context = BrowserContext(
                    browser=browser,
                    sources=loaded_sources,
                    exchanges=loaded_exchanges
                )
                logger.info("Browser context created successfully")
                # potentially add pages to the pool here
                yield context
            except Exception as e:
                logger.error(
                    "Error initializing browser: %s", str(e), exc_info=True)
                raise
            finally:
                # Cleanup
                if browser:
                    logger.info("Closing browser...")
                    await browser.close()
                    logger.info("Browser closed successfully")
    except Exception as e:
        logger.error("Error in browser_lifespan: %s", str(e), exc_info=True)
        raise


async def fetch_source_content(
    source_key: str,
    symbol: str = None,
    company: str = None,
    exchange: str = None,
    normalize: bool = True,
    **kwargs
) -> dict:
    """
    Helper function to fetch and normalize content from any configured source.

    Args:
        source_key: The key of the source in sources.yaml
        symbol: The stock symbol to look up
        company: Optional company name (required for some sources)
        exchange: Optional exchange code (required for some sources)
        **kwargs: Additional parameters that might be needed for the URL template

    Returns:
        dict: Dictionary containing the URL, status, message, and normalized content
    """

    MAXLENGTH = 999999
    try:
        ctx = mcp.get_context()
        browser_ctx = ctx.request_context.lifespan_context
        # get extractor
        extractor = StockDataExtractor(browser_ctx)

        if source_key not in browser_ctx.sources:
            raise ValueError(
                f"Source '{source_key}' not found in configuration")

        source_config = browser_ctx.sources[source_key]
        page = None

        # Prepare template variables
        template_vars = {
            'symbol': symbol or '',
            'company': company or '',
            'exchange': exchange or '',
            **kwargs
        }

        # map exchange
        # TODO: make this a member function map_exchange
        logger.info("Applying exchange mappings for %s", source_key)
        if hasattr(source_config, 'exchange_mappings') and exchange:
            mapping = browser_ctx.exchanges.get(source_key)
            logger.info("Exchange mappings found : %s", str(mapping))
            template_vars['exchange'] = mapping.get(exchange, exchange)

        # Format URL with template variables
        clean_dict = {k: v for k, v in template_vars.items() if v is not None}
        required_params = getattr(source_config, 'required_params', [])
        if any(param not in clean_dict for param in required_params):
            raise ValueError(f"All required template variables ({required_params}) "
                             f"were not found in {clean_dict}")
        url = source_config.url_template.format(**clean_dict)
        logger.info("URL: %s", url)

        # Get a browser page using the browser context
        page = await extractor.get_or_create_page()

        # Navigate to the URL with the specified wait strategy
        # TODO: make this a member function fetch_page, return response
        response = await page.goto(url, wait_until=source_config.wait_strategy)
        if not response or response.status != 200:
            return {
                "url": url,
                "status": "error",
                "message": f"Failed to load page (status: {response.status if response else 'unknown'})"
            }

        # Wait for the page to be fully loaded
        wait_strategy = getattr(source_config, 'wait_strategy', 'networkidle')
        await page.wait_for_load_state(wait_strategy)

        # Get page content
        content = await page.content()
        if not content:
            return {
                "url": url,
                "status": "error",
                "message": "Failed to get page content"
            }

        # TODO: make this a member function get_normalized_page_content, return normalized_text
        # Save to temp file for normalization
        if normalize:
            temp_file = Path(
                f"data/stock_page_{source_key}_{int(time.time())}.html")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(content, encoding='utf-8')
            # Normalize content
            content = normalize_html(temp_file)
            # Cleanup temp file
            temp_file.unlink()

        # truncate content to MAXLENGTH
        if len(content) > MAXLENGTH:
            content = content[:MAXLENGTH]

        return {
            "url": url,
            "status": "success",
            "message": f"Successfully loaded {source_config.name}",
            "content": content,
            "source": source_key
        }

    except Exception as e:
        return {
            "url": url if 'url' in locals() else None,
            "status": "error",
            "message": str(e),
            "source": source_key
        }
    finally:
        # Return the page to the pool
        if page:
            try:
                await extractor.return_page(page)
            except Exception as e:
                print(
                    f"Error returning page to pool in finally block: {str(e)}")

# Initialize MCP server
mcp = FastMCP("stock-symbol-server", lifespan=browser_lifespan)

# image tool opens stockcharts page and saves image locally and returns the image url


# chart display tool shows an image based on a URL


def fn_get_10k_item_from_symbol(symbol, item="1"):
    """
    Get item 1 (or other number) of the latest 10-K annual report filing for a given symbol.

    Args:
        symbol (str): The symbol of the equity.
        item (str):   The item number to return.

    Returns:
        str: The item requested from the latest 10-K annual report filing, or None if not found.

    """

    item_text = ""
    try:
        logger.info("Getting 10-K Item 1 for %s", symbol)
        dl = Downloader(os.getenv("SEC_FIRM"), os.getenv("SEC_USER"))
        html = dl.get_filing_html(ticker=symbol, form="10-K")
        logger.info("HTML length: %d characters", len(html))
        elements = sp.Edgar10QParser().parse(html)
        tree = sp.TreeBuilder().build(elements)
        # look for e.g. "Item 1."
        # sections = [n for n in tree.nodes if re.match(r"^ITEM 1[A-Z]?\.", n.text.strip().upper())]
        sections = [n for n in tree.nodes if re.match(
            r"^ITEM\s+" + item, n.text.strip().upper())]
        logger.info("Sections: %d", len(sections))
        if len(sections) == 0:
            return ""
        item_node = sections[0]
        item_text = item_node.text + "\n\n" + \
            "\n".join([n.text for n in item_node.get_descendants()])
        logger.info("Item text: %d characters", len(item_text))
    except Exception as e:
        logger.info("Error getting 10-K item: %s", e)
    return item_text


@mcp.tool()
async def fetch_url_content(
    url: Annotated[
        str,
        {
            "description": "The URL to fetch content from",
            "example": "https://www.google.com"
        },
    ],
) -> str:
    """Get content from a URL. Use to fetch permissioned content where the password is saved in the Firefox profile."""
    try:
        # TODO: make this a member function in extractor class, use this for all fetches
        # TODO: wrap a function that takes url, wait_strategy, normalize and returns content
        wait_strategy = "networkidle"
        normalize = True
        MAXLENGTH = 999999

        logger.info("URL: %s", url)

        ctx = mcp.get_context()
        browser_ctx = ctx.request_context.lifespan_context
        # get extractor
        extractor = StockDataExtractor(browser_ctx)

        # Get a browser page using the browser context
        page = await extractor.get_or_create_page()

        # Navigate to the URL with the specified wait strategy
        response = await page.goto(url, wait_until=wait_strategy)
        if not response or response.status != 200:
            return {
                "url": url,
                "status": "error",
                "message": f"Failed to load page (status: {response.status if response else 'unknown'})"
            }

        # Wait for the page to be fully loaded, possibly redundant
        await page.wait_for_load_state(wait_strategy)

        # Get page content
        content = await page.content()
        if not content:
            return {
                "url": url,
                "status": "error",
                "message": "Failed to get page content"
            }

        # TODO: make this a member function get_normalized_page_content, return normalized_text
        # Save to temp file for normalization
        if normalize:
            temp_file = Path(
                f"data/stock_page_{secrets.token_hex(20)}.html")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(content, encoding='utf-8')
            # Normalize content
            content = normalize_html(temp_file)
            # Cleanup temp file
            temp_file.unlink()

        return {
            "url": url,
            "status": "success",
            "content": content[:MAXLENGTH],
        }
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "message": str(e)
        }
    finally:
        # Return the page to the pool
        if page:
            try:
                await extractor.return_page(page)
            except Exception as e:
                print(
                    f"Error returning page to pool in finally block: {str(e)}")


@mcp.tool()
def fetch_10k_item1(symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ],
) -> str:
    """Get item 1 of the latest 10-K annual report filing for a given symbol."""
    return fn_get_10k_item_from_symbol(symbol)


@mcp.tool()
def fetch_peers(symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ],
) -> str:
    """Get industry comparables or peers for a stock symbol."""
    obb.account.login(email=os.environ['OPENBB_USER'],
                      password=os.environ['OPENBB_PW'], remember_me=True)

    obj = obb.equity.compare.peers(symbol=symbol, provider='fmp')
    peers = obj.results
    retstr = ""
    for peer in peers.peers_list:
        try:
            obj = obb.equity.profile(peer)
            results = obj.results[0]
            desc = ""
            if results.short_description:
                desc = results.short_description
            elif results.long_description:
                desc = results.long_description

            retstr += f"""
    Symbol:        {peer}
    Name:          {results.name}
    Country:       {results.hq_country}
    Industry:      {results.industry_category}
    Description:   {desc}
    """
        except Exception:
            continue
    return retstr

# https://sethhobson.com/2025/01/building-a-stock-analysis-server-with-mcp-part-1/


class MarketData:
    """Handles all market data fetching operations."""

    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not found in environment")

        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Token {self.api_key}"}

    async def get_historical_data(self, symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        """
        Fetch historical daily data for a given symbol.

        Args:
            symbol (str): The stock symbol to fetch data for.
            lookback_days (int): Number of days to look back from today.

        Returns:
            pd.DataFrame: DataFrame containing historical market data.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            Exception: For other unexpected issues during the fetch operation.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        url = (
            f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?"
            f'startDate={start_date.strftime("%Y-%m-%d")}&'
            f'endDate={end_date.strftime("%Y-%m-%d")}'
        )

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 404:
                        raise ValueError(f"Symbol not found: {symbol}")
                    response.raise_for_status()
                    data = await response.json()

            if not data:
                raise ValueError(f"No data returned for {symbol}")

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            df[["open", "high", "low", "close"]] = df[[
                "adjOpen", "adjHigh", "adjLow", "adjClose"]].round(2)
            df["volume"] = df["adjVolume"].astype(int)
            df["symbol"] = symbol.upper()

            return df

        except aiohttp.ClientError as e:
            raise ConnectionError(
                f"Network error while fetching data for {symbol}: {e}") from e
        except ValueError as ve:
            raise ve  # Propagate value errors (symbol issues, no data, etc.)
        except Exception as e:
            raise Exception(
                f"Unexpected error fetching data for {symbol}: {e}") from e


class TechnicalAnalysis:
    """Technical analysis toolkit with improved performance and readability."""

    @staticmethod
    def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add a core set of technical indicators."""
        try:
            # Adding trend indicators
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)
            df["sma_200"] = ta.sma(df["close"], length=200)

            # Adding volatility indicators and volume
            daily_range = df["high"].sub(df["low"])
            adr = daily_range.rolling(window=20).mean()
            df["adrp"] = adr.div(df["close"]).mul(100)
            df["avg_20d_vol"] = df["volume"].rolling(window=20).mean()

            # Adding momentum indicators
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["rsi"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)

            return df

        except KeyError as e:
            raise KeyError(
                f'Missing column in input DataFrame: {str(e)}') from e
        except Exception as e:
            raise Exception(f'Error calculating indicators: {str(e)}') from e

    @staticmethod
    def check_trend_status(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the current trend status."""
        if df.empty:
            raise ValueError(
                "DataFrame is empty. Ensure it contains valid data.")

        latest = df.iloc[-1]
        return {
            "above_20sma": latest["close"] > latest["sma_20"],
            "above_50sma": latest["close"] > latest["sma_50"],
            "above_200sma": latest["close"] > latest["sma_200"],
            "20_50_bullish": latest["sma_20"] > latest["sma_50"],
            "50_200_bullish": latest["sma_50"] > latest["sma_200"],
            "rsi": latest["rsi"],
            "macd_bullish": latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0),
        }


@mcp.tool()
async def fetch_technical_analysis(symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ],
) -> str:
    """Get technical analysis for a given symbol."""
    market_data = MarketData()
    tech_analysis = TechnicalAnalysis()
    tadf = await market_data.get_historical_data(symbol)
    tadf = tech_analysis.add_core_indicators(tadf)
    trend = tech_analysis.check_trend_status(tadf)
    analysis = f"""
    Technical Analysis for {symbol}:

    Trend Analysis:
    - Above 20 SMA: {'✅' if trend['above_20sma'] else '❌'}
    - Above 50 SMA: {'✅' if trend['above_50sma'] else '❌'}
    - Above 200 SMA: {'✅' if trend['above_200sma'] else '❌'}
    - 20/50 SMA Bullish Cross: {'✅' if trend['20_50_bullish'] else '❌'}
    - 50/200 SMA Bullish Cross: {'✅' if trend['50_200_bullish'] else '❌'}

    Momentum:
    - RSI (14): {trend['rsi']:.2f}
    - MACD Bullish: {'✅' if trend['macd_bullish'] else '❌'}

    Latest Price: ${tadf['close'].iloc[-1]:.2f}
    Average True Range (14): {tadf['atr'].iloc[-1]:.2f}
    Average Daily Range Percentage: {tadf['adrp'].iloc[-1]:.2f}%
    Average Volume (20D): {tadf['avg_20d_vol'].iloc[-1]}
    """
    return analysis

    # these don't show the image inline  in Claude desktop :(
    # you can send image bytes back but you have to expand tool response to see the image
    # @mcp.tool()
    # def return_relative_file_path():
    #     relative_path = os.path.relpath("charts/chart.png")
    #     return {
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "source": {
    #                     "type": "url",
    #                     "url": f"file://{relative_path}"
    #                 }
    #             }
    #         ]
    #     }

    # @mcp.tool()
    # def return_absolute_file_path():
    #     absolute_path = os.path.abspath(
    #         "/Users/drucev/projects/windsurf/MCP/charts/chart.png")
    #     return {
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "source": {
    #                     "type": "url",
    #                     "url": f"file://{absolute_path}"
    #                 }
    #             }
    #         ]
    #     }

    # @Server.list_resources()
    # async def list_resources():
    #     return [
    #         Resource(
    #             uri="file://charts/chart.png",
    #             name="Chart Image",
    #             mimeType="image/png"
    #         )
    #     ]

    # @Server.read_resource()
    # async def read_resource(uri: str):
    #     if uri == "file://charts/chart.png":
    #         with open("charts/chart.png", "rb") as f:
    #             return base64.b64encode(f.read()).decode()


@mcp.tool()
def make_stock_chart(symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ],
) -> MCPImage:
    """Return a stock chart for a given symbol."""

    # Download weekly data
    symbol_df = yf.download(symbol, interval="1wk", period="4y")
    symbol_df.columns = [col[0] if col[1] == symbol else col[0]
                         for col in symbol_df.columns]
    spx_df = yf.download("^GSPC", interval="1wk", period="4y")
    spx_df.columns = [col[0] if col[1] == '^GSPC' else col[0]
                      for col in spx_df.columns]

    # Compute moving averages
    symbol_df['MA13'] = symbol_df['Close'].rolling(window=13).mean()
    symbol_df['MA52'] = symbol_df['Close'].rolling(window=52).mean()

    # Compute relative strength vs SPX
    relative = symbol_df['Close'] / spx_df['Close']
    symbol_df['Rel_SPX'] = relative

    # Create figure with secondary y-axis in the first row
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{}]],  # row 1 has secondary y-axis
        subplot_titles=[
            f"{symbol} Price with Moving Averages & Volume", f"{symbol} Relative to S&P 500"]
    )

    # --- Row 1: Price Candlesticks & MAs (primary y-axis) ---
    fig.add_trace(go.Candlestick(
        x=symbol_df.index,
        open=symbol_df['Open'],
        high=symbol_df['High'],
        low=symbol_df['Low'],
        close=symbol_df['Close'],
        name=symbol,
        increasing_line_color='black',
        decreasing_line_color='red'
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=symbol_df.index,
        y=symbol_df['MA13'],
        mode='lines',
        name='13-week MA',
        line=dict(color='blue')
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=symbol_df.index,
        y=symbol_df['MA52'],
        mode='lines',
        name='52-week MA',
        line=dict(color='orange')
    ), row=1, col=1, secondary_y=False)
    # --- Row 1: Volume on right axis (secondary y-axis) ---
    fig.add_trace(go.Bar(
        x=symbol_df.index,
        y=symbol_df['Volume'],
        name='Volume',
        marker_color='rgba(0, 128, 0, 0.4)',
        showlegend=False
    ), row=1, col=1, secondary_y=True)

    # --- Row 2: Relative to SPX ---
    fig.add_trace(go.Scatter(
        x=symbol_df.index,
        y=symbol_df['Rel_SPX'],
        name=symbol + ' / SPX',
        mode='lines',
        line=dict(color='black')
    ), row=2, col=1)

    # Layout adjustments
    fig.update_layout(
        title=symbol +
        ' Weekly Chart with MAs, Volume (Right Axis), and Relative Strength',
        height=800,
        xaxis=dict(rangeslider_visible=False),
        showlegend=True,
        # Add some margin for better display
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=f"{symbol} / SPX", row=2, col=1)

    # Convert the plot to a PNG image and return as FastMCP Image
    dt = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
    filename = f"charts/{symbol}_{dt}.png"
    fig.write_image(filename, width=800, height=600, scale=2)

    img_bytes = pio.to_image(fig, format="png", width=800, height=600)
    # Encode as base64
    # img_base64 = base64.b64encode(img_bytes).decode()

    # Return the relative file path
    return MCPImage(data=img_bytes, format="png")


@mcp.tool()
async def fetch_image_from_url(
    url: Annotated[
        str,
        {
            "description": "The URL of the image",
            "example": "https://example.com/image.jpg"
        }
    ],
) -> MCPImage:
    """Show an image based on a URL. The URL should point directly to an image."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check if the response is actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(
                    f"URL does not point to an image. Content-Type: {content_type}")

            image_data = response.content

            # Try to open with PIL to validate it's a valid image
            try:
                with PILImage.open(io.BytesIO(image_data)) as img:
                    # Convert to RGB if necessary and save as PNG
                    if img.mode in ('RGBA', 'LA', 'P'):
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                    elif img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')

                    # Save as PNG
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    processed_data = buffer.getvalue()

            except Exception as e:
                raise ValueError(f"Invalid image data: {e}") from e

            return MCPImage(data=processed_data, format="png")

    except httpx.TimeoutException as e:
        raise ValueError("Timeout while fetching image from URL") from e
    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"HTTP error {e.response.status_code} while fetching image") from e
    except Exception as e:
        raise ValueError(f"Error fetching image: {str(e)}") from e

# TODO: news landing page tools should use a pattern and just return the urls of the news stories


@mcp.tool()
async def get_bloomberg_news(
    company: Annotated[
        str,
        {
            "description": "The name of the company",
            "example": "Tesla"
        }
    ],
) -> dict:
    """Search Bloomberg headlines for a given company name."""
    return await fetch_source_content("bloomberg", company=company)


@mcp.tool()
async def get_reuters_news(
    company: Annotated[
        str,
        {
            "description": "The name of the company",
            "example": "Tesla"
        }
    ]
) -> dict:
    """Search Reuters headlines for a given company name."""
    return await fetch_source_content("reuters", company=company)


@mcp.tool()
async def get_yahoo_news(
    symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ]
) -> dict:
    """Search Yahoo Finance headlines for a given stock symbol."""
    return await fetch_source_content("yahoo_news", symbol=symbol)


@mcp.tool()
async def get_yahoo_stats(
    symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "GOOG"
        }
    ]
) -> dict:
    """Search Yahoo Finance for key fundamental statistics for a given stock symbol."""
    return await fetch_source_content("yahoo_stats", symbol=symbol)


# @mcp.tool()
# async def get_ft_news(
#     company: Annotated[
#         str,
#         {
#             "description": "The name of the company",
#             "example": "Goldman Sachs"
#         }
#     ]
# ) -> dict:
#     """Search Financial Times headlines for a given company name."""
#     return await fetch_source_content("ft", company=company)


# @mcp.tool()
# async def get_barrons_news_url(
#     symbol: Annotated[
#         str,
#         {
#             "description": "The stock symbol",
#             "example": "TSLA"
#         }
#     ],
#     company: Annotated[
#         str,
#         {
#             "description": "The name of the company",
#             "example": "Tesla Inc"
#         }
#     ]
# ) -> dict:
#     """Search Barron's headlines for a given stock symbol and company name."""
#     return await fetch_source_content("barrons", symbol=symbol, company=company)


@mcp.tool()
async def get_business_insider_news(
    company: Annotated[
        str,
        {
            "description": "The name of the company",
            "example": "Netflix"
        }
    ]
) -> dict:
    """Search Business Insider headlines for a given company name."""
    return await fetch_source_content("business_insider", company=company)


@mcp.tool()
async def get_google_finance_news(
    symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "META"
        }
    ]
) -> dict:
    """Search Google Finance for a given stock symbol."""
    return await fetch_source_content("google_finance", symbol=symbol)

# # getting timeouts on morningstar, either needs debugging or they are blocking Playwright somehow
# @mcp.tool()
# async def get_morningstar_research(
#     symbol: Annotated[
#         str,
#         {
#             "description": "The stock symbol",
#             "example": "AAPL"
#         }
#     ],
#     exchange: Annotated[
#         str,
#         {
#             "description": "The exchange where the stock is traded",
#             "example": "NASDAQ"
#         }
#     ]
# ) -> dict:
#     """Search Morningstar for a given stock symbol and exchange."""
#     return await fetch_source_content("morningstar", symbol=symbol, exchange=exchange)


@mcp.tool()
async def get_finviz_news(
    symbol: Annotated[
        str,
        {
            "description": "The stock symbol",
            "example": "AAPL"
        }
    ]
) -> dict:
    """Search Finviz for a given stock symbol."""
    return await fetch_source_content("finviz", symbol=symbol)


# @mcp.tool()
# async def get_whalewisdom_updates(
#     symbol: Annotated[
#         str,
#         {
#             "description": "The stock symbol",
#             "example": "AAPL"
#         }
#     ]
# ) -> dict:
#     """Search WhaleWisdom investor transaction updates for a given stock symbol."""
#     return await fetch_source_content("whalewisdom", symbol=symbol)


# @mcp.tool()
# async def get_gurufocus_news(
#     symbol: Annotated[
#         str,
#         {
#             "description": "The stock symbol",
#             "example": "AAPL"
#         }
#     ]
# ) -> dict:
#     """Search GuruFocus investor transaction updates for a given stock symbol."""
#     return await fetch_source_content("gurufocus", symbol=symbol)


@mcp.tool()
async def get_reddit_news(
    company: Annotated[
        str,
        {
            "description": "The name of the company",
            "example": "Tesla"
        }
    ]
) -> dict:
    """Search Reddit for a given company name."""
    return await fetch_source_content("reddit", company=company)

# @mcp.tool()
# def test_params(symbol: str, company: str, exchange: str) -> str:
#     """test tool, returns the parameters received.

#     Args:
#         symbol: The stock symbol (e.g., AAPL, MSFT)
#         company: The company name (e.g., Apple Inc.)
#         exchange: The exchange where the stock is traded (e.g., NASDAQ, NYSE)

#     Returns:
#         A formatted string with the received parameters
#     """
#     # Print parameters to console for debugging (as requested)
#     print("Received parameters:", file=sys.stderr)
#     print(f"  Symbol: {symbol}", file=sys.stderr)
#     print(f"  Company: {company}", file=sys.stderr)
#     print(f"  Exchange: {exchange}", file=sys.stderr)

#     # Return the symbols received (as requested)
#     result = "Stock Symbol Information:\n"
#     result += f"Symbol: {symbol}\n"
#     result += f"Company: {company}\n"
#     result += f"Exchange: {exchange}"

#     return result


def main():
    """Main entry point for the MCP server."""
    # Load and display available sources
    loaded_sources, loaded_exchanges = load_sources_config()

    logger.info("MCP Server - Stock Research")
    logger.info("==========================")
    logger.info("\nAvailable sources:")
    for source_id, source in loaded_sources.items():
        logger.info("- %s (ID: %s)", source.name, source_id)
        logger.info("  Data type: %s", source.data_type)
        logger.info("  Required params: %s", ", ".join(source.required_params))

    logger.info("\nAvailable exchange mappings:")
    for exchange_name, mapping in loaded_exchanges.items():
        logger.info("- %s: %d mappings", exchange_name, len(mapping))

    # Run the server using stdio transport
    # Start the server using stdio transport and a browser context
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()


# TODO: # test and review what comes back. s
# set the tool description
#
# implement the other tools
