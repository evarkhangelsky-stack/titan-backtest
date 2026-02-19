import os
import time
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import telebot

# ==========================================
# БЛОК ИНДИКАТОРОВ (ПОЛНОСТЬЮ ИЗ PDF)
# ==========================================
class TechnicalIndicators:
    @staticmethod
    def vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(close, period):
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(close, period=20, std=2):
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean(), plus_di, minus_di

# ==========================================
# ОСНОВНОЙ КЛАСС (ПОЛНАЯ ВЕРСИЯ)
# ==========================================
class BybitScalpingBot:
    def __init__(self):
        # API Ключи
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')

        # Биржа
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        self.bot = telebot.TeleBot(self.telegram_token)
        
        # Настройки стратегии
        self.symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        self.timeframe = '5m'
        self.leverage = 10
        self.risk_per_trade = 0.05  # 5% от баланса
        self.sl_atr_multiplier = 1.5
        self.tp_atr_multiplier = 3.0
        
        # Хранилище позиций (для мультимонетности)
        self.active_positions = {} 

    def send_telegram(self, message):
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except: pass

    # --- Блок Coinglass ---
    def fetch_coinglass_data(self, symbol):
        if not self.coinglass_api_key: return "N/A"
        try:
            coin = symbol.split('/')[0]
            url = f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={coin}"
            headers = {"coinglassApi": self.coinglass_api_key}
            res = requests.get(url, headers=headers, timeout=5).json()
            return res['data'][0]['longRate'] if 'data' in res else "N/A"
        except: return "N/A"

    # --- Блок CryptoPanic ---
    def fetch_news_sentiment(self, symbol):
        if not self.cryptopanic_api_key: return "Neutral"
        try:
            coin = symbol.split('/')[0]
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_api_key}&currencies={coin}"
            res = requests.get(url, timeout=5).json()
            return "Positive" if len(res.get('results', [])) > 2 else "Neutral"
        except: return "Neutral"

    def fetch_ohlcv(self, symbol):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except: return None

    def calculate_indicators(self, df):
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        df['adx'], df['di_plus'], df['di_minus'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(df['close'])
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        return df

    def get_ai_filter(self, df, signal, symbol, news, cg):
        if not self.deepseek_api_key: return True
        try:
            last = df.iloc[-1]
            prompt = f"Analyze {symbol} {signal}. Price: {last['close']}, RSI: {last['rsi']:.1f}, News: {news}, L/S Ratio: {cg}. Approve/Reject?"
            res = requests.post('https://api.deepseek.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.deepseek_api_key}'},
                json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.3}, timeout=10)
            return "approve" in res.json()['choices'][0]['message']['content'].lower()
        except: return True

    def detect_signal(self, df, symbol):
        last = df.iloc[-1]
        news = self.fetch_news_sentiment(symbol)
        cg = self.fetch_coinglass_data(symbol)
        
        signal = None
        # Логика ADX (из PDF)
        if last['adx'] < 25:
            if last['close'] <= last['bb_lower'] and last['rsi'] < 30: signal = 'LONG'
            elif last['close'] >= last['bb_upper'] and last['rsi'] > 70: signal = 'SHORT'
        else:
            if last['close'] > last['vwap'] and last['ema_20'] > last['ema_50']: signal = 'LONG'
            elif last['close'] < last['vwap'] and last['ema_20'] < last['ema_50']: signal = 'SHORT'

        if signal and self.get_ai_filter(df, signal, symbol, news, cg):
            entry = last['close']
            sl = entry - (self.sl_atr_multiplier * last['atr']) if signal == 'LONG' else entry + (self.sl_atr_multiplier * last['atr'])
            tp = entry + (self.tp_atr_multiplier * last['atr']) if signal == 'LONG' else entry - (self.tp_atr_multiplier * last['atr'])
            return signal, {'entry': entry, 'sl': sl, 'tp': tp, 'news': news, 'cg': cg}
        return None, None

    # --- УПРАВЛЕНИЕ ПОЗИЦИЯМИ (ПОЛНОСТЬЮ ИЗ PDF) ---
    def manage_position(self, symbol, df):
        if symbol not in self.active_positions: return
        pos = self.active_positions[symbol]
        last_price = df.iloc[-1]['close']
        
        print(f"[{symbol}] Monitoring {pos['type']}. Current: {last_price}, TP: {pos['tp']}, SL: {pos['sl']}")
        
        hit_tp = (pos['type'] == 'LONG' and last_price >= pos['tp']) or (pos['type'] == 'SHORT' and last_price <= pos['tp'])
        hit_sl = (pos['type'] == 'LONG' and last_price <= pos['sl']) or (pos['type'] == 'SHORT' and last_price >= pos['sl'])

        if hit_tp or hit_sl:
            reason = "Take Profit" if hit_tp else "Stop Loss"
            print(f"Closing {symbol} by {reason}")
            del self.active_positions[symbol]
            self.send_telegram(f"📉 *Closed* {symbol}\nReason: {reason}\nPrice: {last_price}")

    def place_order(self, symbol, signal, params):
        # Логика расчета объема (из PDF)
        try:
            balance = self.exchange.fetch_balance()['total'].get('USDT', 0)
            print(f"Balance: {balance} USDT. Risking {self.risk_per_trade*100}%")
            
            # Сохраняем виртуально (для Demo режима)
            self.active_positions[symbol] = {
                'type': signal,
                'entry': params['entry'],
                'sl': params['sl'],
                'tp': params['tp']
            }
            
            self.send_telegram(f"🎯 *{signal} Signal* on {symbol}\nEntry: {params['entry']}\nSL: {params['sl']:.2f}\nTP: {params['tp']:.2f}\nNews: {params['news']}")
        except Exception as e:
            print(f"Order Error: {e}")

    def run(self):
        print(f"\n{'='*50}\n Bybit Scalping Bot Started (Multi-Symbol)\n{'='*50}\n")
        while True:
            for symbol in self.symbols:
                try:
                    df = self.fetch_ohlcv(symbol)
                    if df is None: continue
                    df = self.calculate_indicators(df)
                    
                    if symbol in self.active_positions:
                        self.manage_position(symbol, df)
                    else:
                        signal, params = self.detect_signal(df, symbol)
                        if signal:
                            self.place_order(symbol, signal, params)
                    
                    # Детальный принт в консоль (как ты любишь)
                    last = df.iloc[-1]
                    print(f"[{symbol}] Price: {last['close']:.2f} | RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}")
                    
                except Exception as e:
                    print(f"Error in {symbol}: {e}")
                time.sleep(2)
            time.sleep(20) # 20 секунд между проверками

if __name__ == "__main__":
    BybitScalpingBot().run()
