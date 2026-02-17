import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

class TechnicalIndicators:
    """Собственные технические индикаторы"""
    
    @staticmethod
    def vwap(high, low, close, volume):
        """Calculate VWAP"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def rsi(close, period=14):
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def ema(close, period):
        """Calculate EMA"""
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        """Calculate Bollinger Bands"""
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Calculate ADX"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di


class BybitScalpingBot:
    def __init__(self):
        # API keys from environment
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not all([self.api_key, self.api_secret, self.telegram_token, self.telegram_chat_id]):
            raise ValueError("Missing required environment variables.")
        
        # Initialize Bybit
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        
        # Initialize Telegram
        self.bot = telebot.TeleBot(self.telegram_token)
        
        # Trading parameters
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.timeframe = '5m'
        self.position = None
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5
        
        print(f"[{datetime.now()}] Bot initialized for {self.symbols}")

    def send_telegram(self, message):
        """Send message to Telegram"""
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")

    def check_market(self, symbol):
        """Main check logic for a specific symbol"""
        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < 50:
            return

        df = self.calculate_indicators(df)
        
        # Если позиция уже открыта по ЭТОЙ монете — управляем ей
        # (Для простоты в этой версии управление идет одной общей позицией)
        if self.position:
            self.manage_position(df)
            return

        # Ищем сигнал
        signal, signal_type, params = self.detect_signal(df, symbol)
        
        if signal:
            self.place_order(symbol, signal, params)

    def fetch_ohlcv(self, symbol, limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_orderbook(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)
            total_bids = sum([bid[1] for bid in orderbook['bids']])
            total_asks = sum([ask[1] for ask in orderbook['asks']])
            total = total_bids + total_asks
            return (total_bids / total) * 100 if total > 0 else 50
        except:
            return 50

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        adx, di_plus, di_minus = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bb_upper, bb_middle, bb_lower
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        return df

    def get_ai_filter(self, df, symbol, signal_type):
        if not self.deepseek_api_key: return True
        try:
            last_row = df.iloc[-1]
            prompt = f"Analyze {symbol} {signal_type} at {last_row['close']}. RSI {last_row['rsi']:.1f}. Reply ONLY: Approve or Reject"
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.deepseek_api_key}', 'Content-Type': 'application/json'},
                json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3},
                timeout=10
            )
            return 'approve' in response.json()['choices'][0]['message']['content'].lower()
        except:
            return True

    def detect_signal(self, df, symbol):
        last_row = df.iloc[-1]
        bid_ratio = self.fetch_orderbook(symbol)
        signal = None
        
        # Пример логики (упрощенно)
        if last_row['adx'] > 25:
            if last_row['close'] > last_row['vwap'] and bid_ratio > 60:
                signal = 'LONG'
        
        if signal and self.get_ai_filter(df, symbol, signal):
            atr = last_row['atr']
            params = {
                'entry': last_row['close'],
                'stop_loss': last_row['close'] - (atr * 1.2) if signal == 'LONG' else last_row['close'] + (atr * 1.2),
                'take_profit': last_row['close'] + (atr * 2) if signal == 'LONG' else last_row['close'] - (atr * 2),
                'rsi': last_row['rsi'],
                'adx': last_row['adx'],
                'bid_ratio': bid_ratio
            }
            return signal, "Trend", params
        return None, None, None

    def place_order(self, symbol, signal, params):
        message = f"🎯 *Signal for {symbol}*\nType: {signal}\nPrice: {params['entry']:.2f}"
        print(message)
        self.send_telegram(message)
        # Здесь храним данные позиции (упрощенно для одной активной сделки)
        self.position = {'symbol': symbol, 'side': signal, 'entry': params['entry'], 'stop_loss': params['stop_loss'], 'take_profit': params['take_profit'], 'entry_time': datetime.now(), 'trailing_stop_activated': False}

    def manage_position(self, df):
        # Логика управления позицией... (как в твоем коде, но с учетом текущей монеты)
        pass

    def run(self):
        self.send_telegram("🚀 Бот запущен: BTC & ETH")
        while True:
            for symbol in self.symbols:
                try:
                    print(f"[{datetime.now()}] Checking {symbol}")
                    self.check_market(symbol)
                except Exception as e:
                    print(f"Error {symbol}: {e}")
                time.sleep(2)
            time.sleep(300)

if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
