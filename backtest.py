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
            raise ValueError("Missing required environment variables. Check BYBIT_API_KEY, BYBIT_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        
        # Initialize Bybit
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}  # USDT perpetual
        })
        
        # Initialize Telegram
        self.bot = telebot.TeleBot(self.telegram_token)
        
        # Trading parameters
        self.symbol = 'BTC/USDT:USDT'
        self.timeframe = '5m'
        self.position = None
        self.sl_atr_multiplier = 1.2
        self.tp_atr_multiplier = 2.0
        self.trailing_stop_percent = 0.5  # 50% of profit
        
        print(f"[{datetime.now()}] Bot initialized for {self.symbol} on {self.timeframe}")
        self.send_telegram(f"🤖 Bot started\nSymbol: {self.symbol}\nTimeframe: {self.timeframe}")
    
    def send_telegram(self, message):
        """Send message to Telegram"""
        try:
            self.bot.send_message(self.telegram_chat_id, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Telegram error: {e}")
    
    def fetch_ohlcv(self, limit=1000):
        """Fetch candlestick data from Bybit"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return None
    
    def fetch_orderbook(self):
        """Fetch order book to check bid/ask imbalance"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            total_bids = sum([bid[1] for bid in orderbook['bids']])
            total_asks = sum([ask[1] for ask in orderbook['asks']])
            total = total_bids + total_asks
            
            bid_ratio = (total_bids / total) * 100 if total > 0 else 50
            return bid_ratio
        except Exception as e:
            print(f"Error fetching orderbook: {e}")
            return 50  # neutral
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # VWAP
        df['vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        
        # ADX
        adx, di_plus, di_minus = TechnicalIndicators.adx(df['high'], df['low'], df['close'], period=14)
        df['adx'] = adx
        df['di_plus'] = di_plus
        df['di_minus'] = di_minus
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period=14)
        
        # EMAs
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['ema_200'] = TechnicalIndicators.ema(df['close'], period=200)
        
        return df
    
    def get_ai_filter(self, df, signal_type):
        """Get AI filtering from DeepSeek"""
        if not self.deepseek_api_key:
            return True  # Skip AI if no key provided
        
        try:
            last_row = df.iloc[-1]
            
            prompt = f"""Analyze this trading signal:
Symbol: BTC/USDT
Signal: {signal_type}
Price: ${last_row['close']:.2f}
RSI: {last_row['rsi']:.2f}
ADX: {last_row['adx']:.2f}
Price vs VWAP: {'Above' if last_row['close'] > last_row['vwap'] else 'Below'}
EMA Trend: {'Bullish' if last_row['ema_20'] > last_row['ema_50'] else 'Bearish'}

Reply with ONLY one word: "Approve" or "Reject"
"""
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.deepseek_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.3,
                    'max_tokens': 10
                },
                timeout=10
            )
            
            if response.status_code == 200:
                ai_response = response.json()['choices'][0]['message']['content'].strip().lower()
                approved = 'approve' in ai_response
                print(f"AI Filter: {ai_response} -> {'✅ Approved' if approved else '❌ Rejected'}")
                return approved
            else:
                print(f"AI API error: {response.status_code}")
                return True  # Default to approve if API fails
                
        except Exception as e:
            print(f"AI Filter error: {e}")
            return True  # Default to approve if error
    
    def detect_signal(self, df):
        """Detect trading signals based on scalping strategy"""
        last_row = df.iloc[-1]
        
        price = last_row['close']
        rsi = last_row['rsi']
        adx = last_row['adx']
        vwap = last_row['vwap']
        bb_upper = last_row['bb_upper']
        bb_lower = last_row['bb_lower']
        ema_20 = last_row['ema_20']
        ema_50 = last_row['ema_50']
        atr = last_row['atr']
        
        # Skip if NaN values
        if pd.isna([price, rsi, adx, vwap, atr]).any():
            return None, None, None
        
        # Check orderbook imbalance
        bid_ratio = self.fetch_orderbook()
        
        signal = None
        signal_type = None
        
        # Phase detection
        if adx < 25:
            # Sideways market - trade Bollinger Bands + RSI
            if price <= bb_lower and rsi < 30 and bid_ratio > 60:
                signal = 'LONG'
                signal_type = 'Sideways Bounce'
            elif price >= bb_upper and rsi > 70 and bid_ratio < 40:
                signal = 'SHORT'
                signal_type = 'Sideways Rejection'
        else:
            # Trending market - trade with EMA + VWAP
            if price > vwap and ema_20 > ema_50 and rsi > 40 and rsi < 70 and bid_ratio > 60:
                signal = 'LONG'
                signal_type = 'Trend Follow'
            elif price < vwap and ema_20 < ema_50 and rsi < 60 and rsi > 30 and bid_ratio < 40:
                signal = 'SHORT'
                signal_type = 'Trend Follow'
        
        if signal:
            # AI Filter
            if not self.get_ai_filter(df, signal):
                print(f"❌ Signal {signal} rejected by AI filter")
                return None, None, None
            
            # Calculate SL and TP
            if signal == 'LONG':
                entry = price
                stop_loss = entry - (self.sl_atr_multiplier * atr)
                take_profit = entry + (self.tp_atr_multiplier * atr)
            else:
                entry = price
                stop_loss = entry + (self.sl_atr_multiplier * atr)
                take_profit = entry - (self.tp_atr_multiplier * atr)
            
            return signal, signal_type, {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr,
                'rsi': rsi,
                'adx': adx,
                'bid_ratio': bid_ratio
            }
        
        return None, None, None
    
    def place_order(self, signal, params):
        """Place order on Bybit (DEMO MODE - uncomment for live trading)"""
        try:
            # Calculate position size (example: risk 1% of balance)
            balance = self.get_balance()
            risk_amount = balance * 0.01
            position_size = risk_amount / abs(params['entry'] - params['stop_loss'])
            
            # Round to proper precision
            position_size = round(position_size, 3)
            
            message = f"""
🎯 *Signal Detected*
Type: {signal}
Entry: ${params['entry']:.2f}
Stop Loss: ${params['stop_loss']:.2f}
Take Profit: ${params['take_profit']:.2f}
Size: {position_size} BTC
RSI: {params['rsi']:.1f}
ADX: {params['adx']:.1f}
Orderbook: {params['bid_ratio']:.1f}% bids
"""
            
            print(message)
            self.send_telegram(message)
            
            # UNCOMMENT FOR LIVE TRADING
            # order = self.exchange.create_market_order(
            #     symbol=self.symbol,
            #     side='buy' if signal == 'LONG' else 'sell',
            #     amount=position_size
            # )
            
            # Store position
            self.position = {
                'side': signal,
                'entry': params['entry'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'size': position_size,
                'entry_time': datetime.now(),
                'trailing_stop_activated': False
            }
            
            print(f"✅ Order placed: {signal} {position_size} BTC @ ${params['entry']:.2f}")
            
        except Exception as e:
            print(f"Error placing order: {e}")
            self.send_telegram(f"❌ Order error: {str(e)}")
    
    def get_balance(self):
        """Get USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return 10000  # Demo balance
    
    def manage_position(self, df):
        """Manage open position with trailing stop"""
        if not self.position:
            return
        
        current_price = df.iloc[-1]['close']
        side = self.position['side']
        entry = self.position['entry']
        sl = self.position['stop_loss']
        tp = self.position['take_profit']
        
        # Calculate P&L
        if side == 'LONG':
            pnl_percent = ((current_price - entry) / entry) * 100
            
            # Check stop loss
            if current_price <= sl:
                self.close_position(current_price, 'Stop Loss Hit')
                return
            
            # Check take profit
            if current_price >= tp:
                self.close_position(current_price, 'Take Profit Hit')
                return
            
            # Trailing stop logic
            if pnl_percent > self.trailing_stop_percent * 100:
                if not self.position['trailing_stop_activated']:
                    # Move SL to breakeven
                    self.position['stop_loss'] = entry
                    self.position['trailing_stop_activated'] = True
                    print(f"🎯 Trailing stop activated - SL moved to breakeven")
                    self.send_telegram(f"🎯 Trailing stop activated\nNew SL: ${entry:.2f}")
        
        else:  # SHORT
            pnl_percent = ((entry - current_price) / entry) * 100
            
            if current_price >= sl:
                self.close_position(current_price, 'Stop Loss Hit')
                return
            
            if current_price <= tp:
                self.close_position(current_price, 'Take Profit Hit')
                return
            
            if pnl_percent > self.trailing_stop_percent * 100:
                if not self.position['trailing_stop_activated']:
                    self.position['stop_loss'] = entry
                    self.position['trailing_stop_activated'] = True
                    print(f"🎯 Trailing stop activated - SL moved to breakeven")
                    self.send_telegram(f"🎯 Trailing stop activated\nNew SL: ${entry:.2f}")
    
    def close_position(self, exit_price, reason):
        """Close position"""
        if not self.position:
            return
        
        side = self.position['side']
        entry = self.position['entry']
        size = self.position['size']
        
        # Calculate P&L
        if side == 'LONG':
            pnl = (exit_price - entry) * size
            pnl_percent = ((exit_price - entry) / entry) * 100
        else:
            pnl = (entry - exit_price) * size
            pnl_percent = ((entry - exit_price) / entry) * 100
        
        duration = datetime.now() - self.position['entry_time']
        
        message = f"""
{'✅' if pnl > 0 else '❌'} *Position Closed*
Reason: {reason}
Side: {side}
Entry: ${entry:.2f}
Exit: ${exit_price:.2f}
P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)
Duration: {duration}
"""
        
        print(message)
        self.send_telegram(message)
        
        # UNCOMMENT FOR LIVE TRADING
        # self.exchange.create_market_order(
        #     symbol=self.symbol,
        #     side='sell' if side == 'LONG' else 'buy',
        #     amount=size
        # )
        
        self.position = None
    
    def run(self):
        """Main bot loop"""
        print(f"\n{'='*50}")
        print(f"🚀 Bybit Scalping Bot Started")
        print(f"{'='*50}\n")
        
        while True:
            try:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking market...")
                
                # Fetch data
                df = self.fetch_ohlcv()
                if df is None:
                    print("Failed to fetch data, retrying in 60s...")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Check if we have an open position
                if self.position:
                    self.manage_position(df)
                else:
                    # Look for new signal
                    signal, signal_type, params = self.detect_signal(df)
                    
                    if signal:
                        print(f"📊 Signal found: {signal} ({signal_type})")
                        self.place_order(signal, params)
                    else:
                        print("No signal detected")
                
                # Display current market state
                last = df.iloc[-1]
                print(f"Price: ${last['close']:.2f} | RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}")
                
                # Wait for next candle (5 minutes)
                print(f"Next check in 5 minutes...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\n\n👋 Bot stopped by user")
                self.send_telegram("🛑 Bot stopped")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                self.send_telegram(f"❌ Bot error: {str(e)}")
                time.sleep(60)


if __name__ == "__main__":
    bot = BybitScalpingBot()
    bot.run()
