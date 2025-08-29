import os
import ccxt
import pandas as pd
import numpy as np
import pandas_ta
from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import sqlite3
import datetime
import time
import logging
from scipy import stats
import talib
from typing import Dict, List, Tuple, Optional
import requests
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedCryptoFlashBot")

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
EXCHANGE_SECRET = os.getenv('EXCHANGE_SECRET')

# Validate environment variables
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, EXCHANGE_API_KEY, EXCHANGE_SECRET]):
    logger.error("Missing environment variables. Please check your .env file")
    exit(1)

# Initialize exchange with rate limit handling
try:
    exchange = ccxt.bybit({
        'apiKey': EXCHANGE_API_KEY,
        'secret': EXCHANGE_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True,
        },
        'rateLimit': 100,  # Bybit's main rate limit is 100 requests per second
    })
    # Test connection
    exchange.fetch_balance()
    logger.info("Successfully connected to Bybit exchange")
except Exception as e:
    logger.error(f"Failed to initialize exchange: {e}")
    exit(1)

# Initialize Telegram bot
try:
    telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
    logger.info("Telegram bot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Telegram bot: {e}")
    exit(1)

# Database setup
def init_db():
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     pair TEXT, signal TEXT, entry REAL,
                     take_profit REAL, stop_loss REAL,
                     confidence INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     result TEXT, pnl REAL, strategy TEXT,
                     volume_ratio REAL, volatility REAL, rsi_value REAL,
                     market_phase TEXT, order_flow_score REAL,
                     multi_timeframe_score INTEGER, liquidity_zone TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

init_db()

async def send_telegram_message(message):
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info("Telegram message sent successfully")
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")

def record_trade(pair, signal, entry, take_profit, stop_loss, confidence, strategy, 
                 volume_ratio, volatility, rsi_value, market_phase, order_flow_score, 
                 multi_timeframe_score, liquidity_zone):
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        c.execute('''INSERT INTO trades (pair, signal, entry, take_profit, stop_loss, 
                     confidence, strategy, volume_ratio, volatility, rsi_value,
                     market_phase, order_flow_score, multi_timeframe_score, liquidity_zone)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (pair, signal, entry, take_profit, stop_loss, confidence, strategy, 
                   volume_ratio, volatility, rsi_value, market_phase, order_flow_score,
                   multi_timeframe_score, liquidity_zone))
        conn.commit()
        trade_id = c.lastrowid
        conn.close()
        logger.info(f"Trade recorded with ID: {trade_id}")
        return trade_id
    except Exception as e:
        logger.error(f"Failed to record trade: {e}")
        return None

def fetch_with_retry(symbol, timeframe, limit, max_retries=3):
    """Fetch OHLCV data with retry mechanism for rate limiting"""
    for i in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except ccxt.RateLimitExceeded as e:
            wait_time = (2 ** i) * 1000  # Exponential backoff in milliseconds
            logger.warning(f"Rate limit exceeded, retrying in {wait_time/1000} seconds...")
            time.sleep(wait_time / 1000)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    return None

def calculate_advanced_indicators(df):
    try:
        # Price action indicators
        df['ema_8'] = df.ta.ema(length=8)
        df['ema_21'] = df.ta.ema(length=21)
        df['ema_50'] = df.ta.ema(length=50)
        df['ema_100'] = df.ta.ema(length=100)
        df['ema_200'] = df.ta.ema(length=200)
        
        # MACD with multiple timeframes
        macd_fast = df.ta.macd(fast=8, slow=21)
        df['macd_fast'] = macd_fast['MACD_8_21_9']
        df['macd_fast_signal'] = macd_fast['MACDs_8_21_9']
        df['macd_fast_hist'] = macd_fast['MACDh_8_21_9']
        
        macd_slow = df.ta.macd(fast=12, slow=26)
        df['macd_slow'] = macd_slow['MACD_12_26_9']
        df['macd_slow_signal'] = macd_slow['MACDs_12_26_9']
        df['macd_slow_hist'] = macd_slow['MACDh_12_26_9']
        
        # RSI with multiple timeframes
        df['rsi_7'] = df.ta.rsi(length=7)
        df['rsi_14'] = df.ta.rsi(length=14)
        df['rsi_21'] = df.ta.rsi(length=21)
        
        # Stochastic RSI
        stoch_rsi = df.ta.stochrsi()
        df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3']
        
        # Bollinger Bands with different settings
        bb_std1 = df.ta.bbands(length=20, std=1)
        df['bb_upper_1'] = bb_std1['BBU_20_1.0']
        df['bb_lower_1'] = bb_std1['BBL_20_1.0']
        
        bb_std2 = df.ta.bbands(length=20, std=2)
        df['bb_upper_2'] = bb_std2['BBU_20_2.0']
        df['bb_middle_2'] = bb_std2['BBM_20_2.0']
        df['bb_lower_2'] = bb_std2['BBL_20_2.0']
        
        # Volume indicators
        df['volume_ma_20'] = df.ta.sma(length=20, close=df['volume'])
        df['volume_ma_50'] = df.ta.sma(length=50, close=df['volume'])
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Volatility indicators
        df['atr_14'] = df.ta.atr(length=14)
        df['atr_7'] = df.ta.atr(length=7)
        
        # ADX for trend strength
        adx = df.ta.adx()
        df['adx'] = adx['ADX_14']
        df['dmp'] = adx['DMP_14']
        df['dmn'] = adx['DMN_14']
        
        # VWAP for institutional levels
        df['vwap'] = df.ta.vwap()
        
        # Ichimoku Cloud
        ichimoku = df.ta.ichimoku()
        df['ichi_conversion'] = ichimoku['ITS_9_26_52']
        df['ichi_base'] = ichimoku['IKS_9_26_52']
        df['ichi_span_a'] = ichimoku['ISA_9_26_52']
        df['ichi_span_b'] = ichimoku['ISB_9_26_52']
        
        # Price position relative to indicators
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        df['price_vs_ema21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        
        # Calculate price slope (momentum)
        df['price_slope'] = df['close'].diff(5) / df['close'].shift(5) * 100
        
        # Advanced indicators for market phase detection
        df['supertrend'] = df.ta.supertrend(length=10, multiplier=3)['SUPERT_10_3.0']
        df['kst'] = df.ta.kst()
        df['obv'] = df.ta.obv()
        
        # Fibonacci Retracement Levels (from recent high to low)
        recent_high = df['high'].rolling(50).max().iloc[-1]
        recent_low = df['low'].rolling(50).min().iloc[-1]
        diff = recent_high - recent_low
        df['fib_0_236'] = recent_high - 0.236 * diff
        df['fib_0_382'] = recent_high - 0.382 * diff
        df['fib_0_5'] = recent_high - 0.5 * diff
        df['fib_0_618'] = recent_high - 0.618 * diff
        df['fib_0_786'] = recent_high - 0.786 * diff
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

def detect_market_phase(df) -> str:
    """Detect the current market phase (accumulation, markup, distribution, markdown)"""
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check if price is above or below key EMAs
        above_ema200 = latest['close'] > latest['ema_200']
        above_ema100 = latest['close'] > latest['ema_100']
        above_ema50 = latest['close'] > latest['ema_50']
        
        # Check EMA alignment for trend
        ema_bullish = latest['ema_8'] > latest['ema_21'] > latest['ema_50'] > latest['ema_100']
        ema_bearish = latest['ema_8'] < latest['ema_21'] < latest['ema_50'] < latest['ema_100']
        
        # Check ADX for trend strength
        strong_trend = latest['adx'] > 25
        weak_trend = latest['adx'] < 20
        
        # Check volume characteristics
        high_volume = latest['volume_ratio'] > 1.2
        low_volume = latest['volume_ratio'] < 0.8
        
        # Determine market phase
        if ema_bullish and strong_trend and high_volume and above_ema200:
            return "MARKUP"
        elif ema_bearish and strong_trend and high_volume and not above_ema200:
            return "MARKDOWN"
        elif not strong_trend and low_volume and abs(latest['price_vs_ema21']) < 2:
            return "ACCUMULATION"
        elif not strong_trend and high_volume and abs(latest['price_vs_ema21']) < 2:
            return "DISTRIBUTION"
        else:
            return "RANGING"
            
    except Exception as e:
        logger.error(f"Error detecting market phase: {e}")
        return "UNKNOWN"

def analyze_order_flow(df) -> float:
    """Analyze order flow dynamics based on price and volume"""
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate buying and selling pressure
        price_change = latest['close'] - prev['close']
        volume_change = latest['volume'] - prev['volume']
        
        # Volume-price relationship
        if price_change > 0 and volume_change > 0:
            volume_price_score = 1.0  # Strong buying
        elif price_change > 0 and volume_change < 0:
            volume_price_score = 0.5  # Weak buying
        elif price_change < 0 and volume_change > 0:
            volume_price_score = -1.0  # Strong selling
        elif price_change < 0 and volume_change < 0:
            volume_price_score = -0.5  # Weak selling
        else:
            volume_price_score = 0  # Neutral
            
        # OBV trend analysis
        obv_trend = 1 if latest['obv'] > df['obv'].rolling(5).mean().iloc[-1] else -1
        
        # Final order flow score (-1 to 1)
        order_flow_score = (volume_price_score * 0.7) + (obv_trend * 0.3)
        
        return max(-1, min(1, order_flow_score))
        
    except Exception as e:
        logger.error(f"Error analyzing order flow: {e}")
        return 0

def identify_liquidity_zones(df) -> str:
    """Identify key liquidity zones based on recent price action"""
    try:
        # Recent support and resistance levels
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Check if price is near key levels
        near_resistance = abs(current_price - recent_high) / recent_high < 0.005
        near_support = abs(current_price - recent_low) / recent_low < 0.005
        
        if near_resistance:
            return "NEAR_RESISTANCE"
        elif near_support:
            return "NEAR_SUPPORT"
        else:
            return "NO_KEY_LEVEL"
            
    except Exception as e:
        logger.error(f"Error identifying liquidity zones: {e}")
        return "UNKNOWN"

def multi_timeframe_confirmation(pair, timeframes=['15m', '1h', '4h']) -> int:
    """Get confirmation from multiple timeframes (returns score from 0-100)"""
    try:
        confirmations = 0
        total_timeframes = len(timeframes)
        
        for tf in timeframes:
            # Fetch data with retry mechanism
            ohlcv = fetch_with_retry(pair, tf, 100)
            if ohlcv is None or len(ohlcv) < 50:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = calculate_advanced_indicators(df)
            
            # Get current price
            ticker = exchange.fetch_ticker(pair)
            current_price = ticker['last']
            
            # Generate basic signal for this timeframe
            signal, _, _, _, _, _, _, _ = generate_signals(df, current_price)
            
            if signal:
                confirmations += 1
                
        # Calculate confirmation score
        confirmation_score = int((confirmations / total_timeframes) * 100)
        return confirmation_score
        
    except Exception as e:
        logger.error(f"Error in multi-timeframe confirmation: {e}")
        return 0

def generate_signals(df, current_price):
    try:
        # Get the latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Market condition filters
        volatility = latest['atr_14'] / current_price
        adx_strength = latest['adx']
        volume_ratio = latest['volume_ratio']
        rsi_14 = latest['rsi_14']
        price_vs_vwap = latest['price_vs_vwap']
        price_vs_ema21 = latest['price_vs_ema21']
        
        # Advanced market analysis
        market_phase = detect_market_phase(df)
        order_flow_score = analyze_order_flow(df)
        liquidity_zone = identify_liquidity_zones(df)
        
        # Enhanced filter 1: Only trade in favorable market conditions
        if volatility < 0.005:  # Too low volatility
            return None, 0, "LOW_VOLATILITY", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        if adx_strength < 20 and market_phase not in ["MARKUP", "MARKDOWN"]:  # Weak trend
            return None, 0, "WEAK_TREND", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        if volume_ratio < 1.0 and order_flow_score < 0.3:  # Below average volume with weak order flow
            return None, 0, "LOW_VOLUME", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        # Filter out choppy market conditions
        if market_phase in ["ACCUMULATION", "DISTRIBUTION", "RANGING"] and abs(order_flow_score) < 0.5:
            return None, 0, "CHOPPY_MARKET", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        # STRATEGY 1: STRONG TREND FOLLOWING WITH MULTIPLE CONFIRMATIONS
        ema_bullish = (latest['ema_8'] > latest['ema_21'] > latest['ema_50'] > latest['ema_100'])
        ema_bearish = (latest['ema_8'] < latest['ema_21'] < latest['ema_50'] < latest['ema_100'])
        
        macd_bullish = (latest['macd_fast'] > latest['macd_fast_signal'] and 
                       latest['macd_slow'] > latest['macd_slow_signal'] and
                       latest['macd_fast_hist'] > 0 and latest['macd_slow_hist'] > 0)
        
        macd_bearish = (latest['macd_fast'] < latest['macd_fast_signal'] and 
                       latest['macd_slow'] < latest['macd_slow_signal'] and
                       latest['macd_fast_hist'] < 0 and latest['macd_slow_hist'] < 0)
        
        price_above_cloud = (current_price > latest['ichi_span_a'] and current_price > latest['ichi_span_b'])
        price_below_cloud = (current_price < latest['ichi_span_a'] and current_price < latest['ichi_span_b'])
        
        # Strong bullish trend with order flow confirmation
        if (ema_bullish and macd_bullish and price_above_cloud and
            45 < rsi_14 < 75 and volume_ratio > 1.2 and
            price_vs_vwap > -0.5 and price_vs_ema21 > -1.0 and
            order_flow_score > 0.7 and market_phase == "MARKUP"):
            
            # Additional confirmation: Price above key Fibonacci level
            if current_price > latest['fib_0_618']:
                confidence = min(95, 75 + (volume_ratio - 1) * 10 + (rsi_14 - 50) / 25 * 10 + order_flow_score * 10)
                return 'BUY', confidence, "STRONG_TREND", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        # Strong bearish trend with order flow confirmation
        if (ema_bearish and macd_bearish and price_below_cloud and
            25 < rsi_14 < 55 and volume_ratio > 1.2 and
            price_vs_vwap < 0.5 and price_vs_ema21 < 1.0 and
            order_flow_score < -0.7 and market_phase == "MARKDOWN"):
            
            # Additional confirmation: Price below key Fibonacci level
            if current_price < latest['fib_0_382']:
                confidence = min(95, 75 + (volume_ratio - 1) * 10 + (50 - rsi_14) / 25 * 10 + abs(order_flow_score) * 10)
                return 'SELL', confidence, "STRONG_TREND", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        # STRATEGY 2: MEAN REVERSION AT KEY LEVELS WITH CONFLUENCE
        bb_touch_upper = (current_price >= latest['bb_upper_2'] * 0.995)
        bb_touch_lower = (current_price <= latest['bb_lower_2'] * 1.005)
        
        # Check if price is at key Fibonacci levels
        at_fib_618 = abs(current_price - latest['fib_0_618']) / latest['fib_0_618'] < 0.005
        at_fib_382 = abs(current_price - latest['fib_0_382']) / latest['fib_0_382'] < 0.005
        
        # Bollinger Band bounce with RSI confirmation at key levels
        if (bb_touch_lower and (at_fib_618 or liquidity_zone == "NEAR_SUPPORT") and 
            rsi_14 < 35 and latest['stoch_rsi_k'] < 20 and 
            latest['stoch_rsi_d'] < 20 and volume_ratio > 1.5 and 
            latest['macd_fast_hist'] > prev['macd_fast_hist'] and
            order_flow_score > 0.5):
            
            confidence = min(90, 65 + (35 - rsi_14) + (volume_ratio - 1) * 10 + order_flow_score * 10)
            return 'BUY', confidence, "MEAN_REVERSION", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        if (bb_touch_upper and (at_fib_382 or liquidity_zone == "NEAR_RESISTANCE") and 
            rsi_14 > 65 and latest['stoch_rsi_k'] > 80 and 
            latest['stoch_rsi_d'] > 80 and volume_ratio > 1.5 and 
            latest['macd_fast_hist'] < prev['macd_fast_hist'] and
            order_flow_score < -0.5):
            
            confidence = min(90, 65 + (rsi_14 - 65) + (volume_ratio - 1) * 10 + abs(order_flow_score) * 10)
            return 'SELL', confidence, "MEAN_REVERSION", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        # STRATEGY 3: BREAKOUT TRADING WITH VOLUME CONFIRMATION
        recent_high = df['high'].rolling(20).max().iloc[-2]  # Previous period high
        recent_low = df['low'].rolling(20).min().iloc[-2]   # Previous period low
        
        breakout_above = current_price > recent_high and volume_ratio > 1.8
        breakout_below = current_price < recent_low and volume_ratio > 1.8
        
        if breakout_above and rsi_14 < 70 and order_flow_score > 0.7 and market_phase in ["ACCUMULATION", "MARKUP"]:
            confidence = min(85, 70 + (volume_ratio - 1) * 8 + order_flow_score * 10)
            return 'BUY', confidence, "BREAKOUT", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
            
        if breakout_below and rsi_14 > 30 and order_flow_score < -0.7 and market_phase in ["DISTRIBUTION", "MARKDOWN"]:
            confidence = min(85, 70 + (volume_ratio - 1) * 8 + abs(order_flow_score) * 10)
            return 'SELL', confidence, "BREAKOUT", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
        return None, 0, "NO_CLEAR_SIGNAL", volume_ratio, volatility, rsi_14, market_phase, order_flow_score, liquidity_zone
        
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return None, 0, "ERROR", 0, 0, 0, "ERROR", 0, "ERROR"

def calculate_tp_sl(current_price, signal, volatility, confidence, rsi_value, order_flow_score, liquidity_zone):
    # Dynamic position sizing based on multiple factors
    base_risk = 0.02  # Base 2% risk per trade
    
    # Adjust risk based on confidence
    if confidence >= 90:
        risk_per_trade = base_risk * 1.5
    elif confidence >= 80:
        risk_per_trade = base_risk * 1.2
    elif confidence >= 70:
        risk_per_trade = base_risk
    else:
        risk_per_trade = base_risk * 0.7
    
    # Adjust risk based on order flow strength
    risk_per_trade *= (1 + abs(order_flow_score) * 0.5)
    
    # Adjust risk based on volatility (lower risk in high volatility)
    volatility_factor = max(0.5, min(2.0, 0.01 / volatility))
    risk_per_trade *= volatility_factor
    
    # Calculate stop loss based on ATR and volatility
    atr_stop_multiplier = 1.5
    
    # Wider stops for mean reversion, tighter for breakouts
    if liquidity_zone in ["NEAR_SUPPORT", "NEAR_RESISTANCE"]:
        atr_stop_multiplier = 2.0  # Wider stops for mean reversion
    
    atr_stop = volatility * current_price * atr_stop_multiplier
    
    # Calculate position size
    stop_loss_pct = atr_stop / current_price
    
    # Adjust stop loss based on RSI (wider stops for extreme RSI values)
    if (signal == 'BUY' and rsi_value < 30) or (signal == 'SELL' and rsi_value > 70):
        stop_loss_pct *= 1.3  # 30% wider stop for extreme conditions
    
    # Calculate take profit based on dynamic risk:reward ratio
    # Higher reward for stronger signals
    base_reward_ratio = 2.5
    reward_ratio = base_reward_ratio + (confidence - 70) / 30  # Scale from 2.5 to 3.5 for 70-100% confidence
    
    if signal == 'BUY':
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + stop_loss_pct * reward_ratio)
        
        # Adjust TP to avoid obvious resistance levels
        if liquidity_zone == "NEAR_RESISTANCE":
            take_profit = current_price * (1 + stop_loss_pct * (reward_ratio * 0.8))
    else:  # SELL
        stop_loss = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - stop_loss_pct * reward_ratio)
        
        # Adjust TP to avoid obvious support levels
        if liquidity_zone == "NEAR_SUPPORT":
            take_profit = current_price * (1 - stop_loss_pct * (reward_ratio * 0.8))
    
    return take_profit, stop_loss

def analyze_pair(pair, timeframe='15m', limit=100):
    try:
        # Fetch OHLCV data with retry mechanism
        ohlcv = fetch_with_retry(pair, timeframe, limit)
        if ohlcv is None or len(ohlcv) < 50:  # Not enough data
            return None, None, None, None, 0, [], 0, 0, 0, "UNKNOWN", 0, 0, "UNKNOWN"
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate indicators
        df = calculate_advanced_indicators(df)
        
        # Get current price
        ticker = exchange.fetch_ticker(pair)
        current_price = ticker['last']
        
        # Generate signals
        signal, confidence, strategy, volume_ratio, volatility, rsi_value, market_phase, order_flow_score, liquidity_zone = generate_signals(df, current_price)
        
        if signal and confidence >= 75:
            # Get multi-timeframe confirmation
            mtf_score = multi_timeframe_confirmation(pair)
            
            # Only proceed if we have strong multi-timeframe confirmation
            if mtf_score >= 60:
                take_profit, stop_loss = calculate_tp_sl(current_price, signal, volatility, confidence, rsi_value, order_flow_score, liquidity_zone)
                return signal, current_price, take_profit, stop_loss, confidence, [strategy], volume_ratio, volatility, rsi_value, market_phase, order_flow_score, mtf_score, liquidity_zone
        
        return None, None, None, None, 0, [], volume_ratio, volatility, rsi_value, market_phase, order_flow_score, 0, liquidity_zone
        
    except Exception as e:
        logger.error(f"Error analyzing {pair}: {e}")
        return None, None, None, None, 0, [], 0, 0, 0, "ERROR", 0, 0, "ERROR"

async def check_trade_results():
    """Check previously sent trades to see if they hit TP or SL"""
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        c.execute("SELECT id, pair, signal, entry, take_profit, stop_loss FROM trades WHERE result IS NULL")
        open_trades = c.fetchall()
        
        for trade in open_trades:
            trade_id, pair, signal, entry, take_profit, stop_loss = trade
            
            try:
                # Respect rate limits
                time.sleep(exchange.rateLimit / 1000)
                
                ticker = exchange.fetch_ticker(pair)
                current_price = ticker['last']
                
                if signal == 'BUY':
                    if current_price >= take_profit:
                        pnl = ((take_profit - entry) / entry) * 100
                        await send_telegram_message(
                            f"✅ <b>TRADE WON</b> ✅\n"
                            f"Pair: {pair}\n"
                            f"Signal: {signal}\n"
                            f"Entry: {entry:.4f}\n"
                            f"TP: {take_profit:.4f}\n"
                            f"Current: {current_price:.4f}\n"
                            f"PnL: +{pnl:.2f}%\n\n"
                            f"<b>#Profit #Trading</b>"
                        )
                        c.execute("UPDATE trades SET result = ?, pnl = ? WHERE id = ?", ("WIN", pnl, trade_id))
                    elif current_price <= stop_loss:
                        pnl = ((stop_loss - entry) / entry) * 100
                        await send_telegram_message(
                            f"❌ <b>TRADE LOST</b> ❌\n"
                            f"Pair: {pair}\n"
                            f"Signal: {signal}\n"
                            f"Entry: {entry:.4f}\n"
                            f"SL: {stop_loss:.4f}\n"
                            f"Current: {current_price:.4f}\n"
                            f"PnL: -{abs(pnl):.2f}%\n\n"
                            f"<b>#Loss #Trading</b>"
                        )
                        c.execute("UPDATE trades SET result = ?, pnl = ? WHERE id = ?", ("LOSS", pnl, trade_id))
                
                elif signal == 'SELL':
                    if current_price <= take_profit:
                        pnl = ((entry - take_profit) / entry) * 100
                        await send_telegram_message(
                            f"✅ <b>TRADE WON</b> ✅\n"
                            f"Pair: {pair}\n"
                            f"Signal: {signal}\n"
                            f"Entry: {entry:.4f}\n"
                            f"TP: {take_profit:.4f}\n"
                            f"Current: {current_price:.4f}\n"
                            f"PnL: +{pnl:.2f}%\n\n"
                            f"<b>#Profit #Trading</b>"
                        )
                        c.execute("UPDATE trades SET result = ?, pnl = ? WHERE id = ?", ("WIN", pnl, trade_id))
                    elif current_price >= stop_loss:
                        pnl = ((entry - stop_loss) / entry) * 100
                        await send_telegram_message(
                            f"❌ <b>TRADE LOST</b> ❌\n"
                            f"Pair: {pair}\n"
                            f"Signal: {signal}\n"
                            f"Entry: {entry:.4f}\n"
                            f"SL: {stop_loss:.4f}\n"
                            f"Current: {current_price:.4f}\n"
                            f"PnL: -{abs(pnl):.2f}%\n\n"
                            f"<b>#Loss #Trading</b>"
                        )
                        c.execute("UPDATE trades SET result = ?, pnl = ? WHERE id = ?", ("LOSS", pnl, trade_id))
            
            except Exception as e:
                logger.error(f"Error checking trade {trade_id}: {e}")
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error in check_trade_results: {e}")

async def send_daily_performance_report():
    """Send daily performance report"""
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        
        # Get today's date
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        c.execute("""\
            SELECT \
                COUNT(*) as total_signals,\
                SUM(CASE WHEN result = \"WIN\" THEN 1 ELSE 0 END) as wins,\
                SUM(CASE WHEN result = \"LOSS\" THEN 1 ELSE 0 END) as losses,\\
                SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) as pending,\
                SUM(CASE WHEN result = \"WIN\" THEN pnl ELSE 0 END) as total_profit,\
                SUM(CASE WHEN result = \"LOSS\" THEN pnl ELSE 0 END) as total_loss\
            FROM trades \
            WHERE date(timestamp) = ?\
        """, (today,))
        
        stats = c.fetchone()
        total, wins, losses, pending, profit, loss = stats
        
        if total > 0:
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            net_profit = profit + loss
            
            # Get best performing pair
            c.execute("""\
                SELECT pair, SUM(CASE WHEN result = \"WIN\" THEN pnl ELSE 0 END) as pair_profit
                FROM trades 
                WHERE date(timestamp) = ?
                GROUP BY pair
                ORDER BY pair_profit DESC
                LIMIT 1
            """, (today,))
            
            best_pair = c.fetchone()
            best_pair_name = best_pair[0] if best_pair else "N/A"
            best_pair_profit = best_pair[1] if best_pair else 0
            
            # Get best performing strategy
            c.execute("""\
                SELECT strategy, SUM(CASE WHEN result = \"WIN\" THEN pnl ELSE 0 END) as strategy_profit
                FROM trades 
                WHERE date(timestamp) = ?
                GROUP BY strategy
                ORDER BY strategy_profit DESC
                LIMIT 1
            """, (today,))
            
            best_strategy = c.fetchone()
            best_strategy_name = best_strategy[0] if best_strategy else "N/A"
            best_strategy_profit = best_strategy[1] if best_strategy else 0
            
            message = (
                f"📊 <b>DAILY PERFORMANCE REPORT - {today}</b>\n"
                f"─────────────────────\n"
                f"Total Signals: {total}\n"
                f"Wins: {wins} 🟢\n"
                f"Losses: {losses} 🔴\n"
                f"Pending: {pending} ⏳\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Total PnL: {net_profit:+.2f}%\n"
                f"Best Performer: {best_pair_name} ({best_pair_profit:+.2f}%)\n"
                f"Best Strategy: {best_strategy_name} ({best_strategy_profit:+.2f}%)\n"
                f"─────────────────────\n"
                f"<b>#DailyReport #TradingBot</b>"
            )
            
            await send_telegram_message(message)
        
        conn.close()
    except Exception as e:
        logger.error(f"Error sending daily report: {e}")

async def monitor_markets():
    # Top liquid pairs with good volatility - focus on majors with high volume
    pairs = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
        'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT',
        'MATIC/USDT', 'LINK/USDT', 'DOGE/USDT', 'ATOM/USDT'
    ]
    
    # Track signals to avoid duplicates
    recent_signals = {}
    signal_cooldown = 3600  # 1 hour cooldown for the same pair
    
    last_report_hour = -1
    
    logger.info("Starting advanced market monitoring...")
    
    while True:
        try:
            current_time = datetime.datetime.utcnow()
            current_hour = current_time.hour
            
            # Send daily report at 00:00 UTC
            if current_hour == 0 and last_report_hour != 0:
                await send_daily_performance_report()
                last_report_hour = 0
            
            # Check trade results every 5 minutes
            if current_time.minute % 5 == 0:
                await check_trade_results()
            
            for pair in pairs:
                # Check if this pair is in cooldown
                if pair in recent_signals:
                    time_since_signal = time.time() - recent_signals[pair]
                    if time_since_signal < signal_cooldown:
                        continue
                
                # Add delay to respect rate limits
                time.sleep(exchange.rateLimit / 1000)
                
                signal, current_price, take_profit, stop_loss, confidence, strategies, volume_ratio, volatility, rsi_value, market_phase, order_flow_score, mtf_score, liquidity_zone = analyze_pair(pair)
                
                if signal and confidence >= 75 and mtf_score >= 60:
                    # Record the signal time
                    recent_signals[pair] = time.time()
                    
                    trade_id = record_trade(pair, signal, current_price, take_profit, stop_loss, confidence, 
                                          '+'.join(strategies), volume_ratio, volatility, rsi_value, 
                                          market_phase, order_flow_score, mtf_score, liquidity_zone)
                    
                    if trade_id:
                        # Create signal message with emojis based on signal type
                        if signal == 'BUY':
                            signal_emoji = "🟢"
                        else:
                            signal_emoji = "🔴"
                            
                        message = (
                            f"{signal_emoji} <b>FLASH SIGNAL</b> {signal_emoji}\n"
                            f"Pair: <b>{pair}</b>\n"
                            f"Signal: <b>{signal}</b>\n"
                            f"Entry: <b>{current_price:.4f}</b>\n"
                            f"Take Profit: <b>{take_profit:.4f}</b>\n"
                            f"Stop Loss: <b>{stop_loss:.4f}</b>\n"
                            f"Confidence: <b>{confidence}%</b>\n"
                            f"Strategy: <b>{', '.join(strategies)}</b>\n"
                            f"Volume Ratio: <b>{volume_ratio:.2f}x</b>\n"
                            f"RSI: <b>{rsi_value:.2f}</b>\n"
                            f"Market Phase: <b>{market_phase}</b>\n"
                            f"Order Flow: <b>{order_flow_score:.2f}</b>\n"
                            f"MTF Score: <b>{mtf_score}%</b>\n"
                            f"Liquidity Zone: <b>{liquidity_zone}</b>\n\n"
                            f"<b>#Crypto #Trading #{pair.replace('/', '')}</b>"
                        )
                        
                        await send_telegram_message(message)
                        logger.info(f"High confidence signal sent for {pair} with confidence {confidence}%")
            
            # Clean up old signals from tracking
            current_time = time.time()
            recent_signals = {k: v for k, v in recent_signals.items() if current_time - v < signal_cooldown}
            
            # Wait before next scan (3 minutes) - adjusted for rate limiting
            await asyncio.sleep(180)
            
        except Exception as e:
            logger.error(f"Error in monitor_markets loop: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying

if __name__ == '__main__':
    print("Starting ADVANCED PROFITABLE crypto flash signal bot...")
    logger.info("Bot started successfully")
    asyncio.run(monitor_markets())
