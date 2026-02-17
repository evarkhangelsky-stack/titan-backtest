import os, requests, numpy as np, pandas as pd, pandas_ta as ta, telebot, time

# --- [CONFIG] ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CG_KEY = os.getenv("COINGLASS_API_KEY")
CP_KEY = os.getenv("CRYPTOPANIC_API_KEY")
bot = telebot.TeleBot(TOKEN)

# --- [БЛОК 1: СБОР ДАННЫХ] ---
class DataCollector:
    def __init__(self, symbol="ETHUSDT"):
        self.symbol = symbol
        self.coin = symbol.replace("USDT", "")

    def get_bybit_market_data(self):
        try:
            url = "https://api.bybit.com/v5/market"
            k_res = requests.get(f"{url}/kline", params={"category": "linear", "symbol": self.symbol, "interval": "5", "limit": 1000}, timeout=10).json()
            klines = k_res['result']['list'][::-1]
            t_res = requests.get(f"{url}/tickers", params={"category": "linear", "symbol": self.symbol}, timeout=10).json()
            ticker = t_res['result']['list'][0]
            o_res = requests.get(f"{url}/orderbook", params={"category": "linear", "symbol": self.symbol, "limit": 50}, timeout=10).json()
            return {"klines": klines, "ticker": ticker, "orderbook": o_res['result']}
        except Exception as e:
            print(f"Ошибка Bybit: {e}"); return None

    def get_coinglass_data(self):
        if not CG_KEY: return None
        try:
            headers = {"accept": "application/json", "CG-API-KEY": CG_KEY}
            res = requests.get(f"https://open-api.coinglass.com/public/v2/long_short?time_type=h1&symbol={self.coin}", headers=headers, timeout=10).json()
            return res.get('data', [{}])[0]
        except Exception as e:
            print(f"Ошибка Coinglass: {e}"); return None

    def get_cryptopanic_news(self):
        if not CP_KEY: return []
        try:
            res = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={CP_KEY}&currencies={self.coin}&kind=news&filter=hot", timeout=10).json()
            return res.get('results', [])[:5]
        except Exception as e:
            print(f"Ошибка News: {e}"); return []

    def collect_all(self):
        return {"market": self.get_bybit_market_data(), "blockchain": self.get_coinglass_data(), "news": self.get_cryptopanic_news()}

# --- [БЛОК 2-3: АНАЛИЗАТОР (ИНДИКАТОРЫ + МАТЕМАТИКА)] ---
class TechnicalAnalyzer:
    def __init__(self, raw_bundle):
        self.market = raw_bundle.get('market')
        
    def prepare_df(self):
        if not self.market: return None
        df = pd.DataFrame(self.market['klines'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 't'])
        for col in ['o', 'h', 'l', 'c', 'v']: 
            df[col] = pd.to_numeric(df[col])
        return df

    def calculate(self):
        df = self.prepare_df()
        if df is None or len(df) < 30: return None # Проверка на достаточность данных
        
        res = {'price': df['c'].iloc[-1]}
        
        # Трендовые
        res['ema20'] = ta.ema(df['c'], length=20).iloc[-1]
        res['ema50'] = ta.ema(df['c'], length=50).iloc[-1]
        res['ema200'] = ta.ema(df['c'], length=200).iloc[-1] if len(df) >= 200 else res['ema50']
        res['vwap'] = (df['v'] * (df['h'] + df['l'] + df['c']) / 3).sum() / df['v'].sum()
        
        # Осцилляторы
        res['rsi'] = ta.rsi(df['c'], length=14).iloc[-1]
        macd = ta.macd(df['c'])
        res['macd_h'] = macd.iloc[-1, 1] # Берем вторую колонку (гистограмму) напрямую
        
        # Волатильность (Исправленный блок)
        bb = ta.bbands(df['c'], length=20, std=2)
        # Вместо имен 'BBU_20_2.0' берем по индексу: 0 - нижняя, 1 - средняя, 2 - верхняя
        res['bb_up'] = bb.iloc[-1, 2]
        res['bb_low'] = bb.iloc[-1, 0]
        
        res['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14).iloc[-1]
        res['adx'] = ta.adx(df['h'], df['l'], df['c'], length=14).iloc[-1, 0]
        
        return res

    def analyze_orderbook(self):
        try:
            if not self.market or 'orderbook' not in self.market: return 0.5
            ob = self.market['orderbook']
            bids = sum([float(i[1]) for i in ob['b']])
            asks = sum([float(i[1]) for i in ob['a']])
            return bids / (bids + asks) if (bids + asks) > 0 else 0.5
        except: return 0.5

# --- [БЛОК 4-5: SMART ANALYST & AI] ---
class SmartAnalyst:
    def __init__(self, tech_data, raw_bundle):
        self.tech, self.blockchain, self.news = tech_data, raw_bundle.get('blockchain'), raw_bundle.get('news')

    def analyze_all(self):
        rep = {'ls_ratio': float(self.blockchain.get('v', 1.0)) if self.blockchain else 1.0}
        bull_w = ['buy', 'pump', 'growth', 'surge', 'bullish', 'support']
        score = 0
        titles = ""
        for n in self.news:
            titles += n['title'] + " | "
            if any(w in n['title'].lower() for w in bull_w): score += 1
        rep['sentiment'] = "Positive" if score > 0 else "Neutral/Negative"
        rep['news_summary'] = titles[:200]
        
        prompt = f"ETH:{self.tech['price']}. RSI:{round(self.tech['rsi'],1)}, Sent:{rep['sentiment']}. Pro assessment 15 words."
        try:
            res = requests.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {DS_KEY}"},
                               json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}, timeout=10).json()
            rep['ai_verdict'] = res['choices'][0]['message']['content']
        except: rep['ai_verdict'] = "AI Offline."
        return rep

# --- [БЛОК 6: ГРАФИКА (ГЛАЗА БОТА)] ---
class ChartGeometry:
    def __init__(self, raw_bundle):
        m = raw_bundle.get('market', {})
        self.klines = m.get('klines', [])
        if self.klines:
            self.c = np.array([float(x[4]) for x in self.klines])
            self.h = np.array([float(x[2]) for x in self.klines])
            self.l = np.array([float(x[3]) for x in self.klines])

    def detect_structure(self):
        if len(self.c) < 50: return "Unknown"
        h, l = max(self.h[-20:-1]), min(self.l[-20:-1])
        if self.c[-1] > h: return "BOS Bullish"
        if self.c[-1] < l: return "BOS Bearish"
        return "Range"

    def find_patterns(self):
        if len(self.c) < 60: return "Neutral"
        h1, h2 = max(self.h[-40:-20]), max(self.h[-20:])
        if abs(h1 - h2) / h1 < 0.002: return "Double Top"
        l1, l2 = min(self.l[-40:-20]), min(self.l[-20:])
        if abs(l1 - l2) / l1 < 0.002: return "Double Bottom"
        return "Neutral"

    def get_sr_levels(self):
        all_p = np.concatenate([self.h[-100:], self.l[-100:]])
        lvls = [round(p, 2) for p in all_p if np.sum(np.abs(all_p - p) / p < 0.001) > 3]
        return sorted(list(set(lvls)))[-3:]

# --- [БЛОК СТРАТЕГИИ] ---
class StrategyManager:
def __init__(self, tech, struct, smart):
        self.t, self.s, self.a = tech, struct, smart
        
def calculate_score(self):
    sc = 0
    price = self.t['price']
    vwap = self.t.get('vwap', price)
    rsi = self.t['rsi']
    adx = self.t['adx']
    bb_low = self.t['bb_low']
    bb_up = self.t['bb_up']
    
    # 1. ПАТТЕРН "ОТКЛОНЕНИЕ ОТ VWAP" (Mean Reversion)
    # Если цена сильно улетела от VWAP — ждем возврат
    if price < vwap * 0.995: # Упали на 0.5% ниже VWAP
        sc += 1
    elif price > vwap * 1.005: # Выросли на 0.5% выше VWAP
        sc -= 1

    # 2. СКАЛЬПИНГ ПО БОЛЛИНДЖЕРУ (В боковике ADX < 25)
    if adx < 25:
        if price <= bb_low and rsi < 30:
            sc += 2  # Локальное дно
        elif price >= bb_up and rsi > 70:
            sc -= 2  # Локальный хай

    # 3. ИМПУЛЬС (Breakout) - если летим с объемами
    if adx > 30:
        if price > vwap and rsi > 60:
            sc += 2  # Входим в разгон тренда
        elif price < vwap and rsi < 40:
            sc -= 2

    # 4. СТАКАН (Orderbook Imbalance)
    ob_ratio = TechnicalAnalyzer(self.t).analyze_orderbook() # Нужен доступ к методу
    if ob_ratio > 0.6: sc += 1 # Покупателей больше
    elif ob_ratio < 0.4: sc -= 1 # Продавцов больше

    return sc

   def generate_setup(self):
        sc = self.calculate_score()
        
        # Пороги входа для скальпинга
        if sc >= 3:
            side = "LONG"
        elif sc <= -3:
            side = "SHORT"
        else:
            return {"side": None} 
            
        atr = self.t.get('atr', 0)
        if atr == 0: return {"side": None}
        
        entry = self.t['price']

        # --- НОВАЯ НАСТРОЙКА ДЛЯ СКАЛЬПИНГА (М5-М15) ---
        # Меньшие множители позволяют забирать быстрые импульсы
        sl_mult = 1.2  # Стоп-лосс: 1.2 * ATR
        tp_mult = 2.0  # Тейк-профит: 2.0 * ATR (соотношение риск/прибыль ~1:1.6)

        if side == "LONG":
            sl = round(entry - (atr * sl_mult), 2)
            tp = round(entry + (atr * tp_mult), 2)
        else: # SHORT
            sl = round(entry + (atr * sl_mult), 2)
            tp = round(entry - (atr * tp_mult), 2)
        
        return {
            "side": side, 
            "entry": entry, 
            "sl": sl, 
            "tp": tp, 
            "score": sc
        }


# --- [ГЛАВНЫЙ БЛОК ЗАПУСКА С УЛУЧШЕННОЙ ГРАФИКОЙ И ТАБЛИЦЕЙ] ---
import matplotlib.pyplot as plt
import io

def run_visual_backtest(symbol="ETHUSDT"):
    collector = DataCollector(symbol)
    raw = collector.get_bybit_market_data() 
    if not raw: return
    
    df = pd.DataFrame(raw['klines'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 't'])
    for col in ['o', 'h', 'l', 'c', 'v']: df[col] = pd.to_numeric(df[col])
    
    trades_log = []
    print(f"🧐 Запуск умного бэктеста {symbol}...")

    last_trade_idx = 0
    cooldown = 15  # Не заходим в сделки слишком часто (защита от "ножей")

    for i in range(200, len(df) - 20):
        # Пропускаем итерацию, если мы в "режиме ожидания" после сделки
        if i < last_trade_idx + cooldown: 
            continue 
            
        temp_bundle = {'market': {'klines': raw['klines'][:i+1]}, 'blockchain': {}, 'news': []}
        
        tech = TechnicalAnalyzer(temp_bundle).calculate()
        if not tech: continue
        
        geo = ChartGeometry(temp_bundle)
        struct = {'structure': geo.detect_structure(), 'patterns': geo.find_patterns()}
        
        # Анализ через StrategyManager
        setup = StrategyManager(tech, struct, {'ls_ratio':1, 'sentiment':'Neutral'}).generate_setup()
        
        if setup.get('side'):
            side, entry, tp, sl = setup['side'], setup['entry'], setup['tp'], setup['sl']
            
            # Проверяем, что случилось с ценой в следующие 20 свечей
            for j in range(i + 1, i + 20):
                h, l = df['h'].iloc[j], df['l'].iloc[j]
                
                res = None
                if side == "LONG":
                    if h >= tp: res = "WIN"
                    elif l <= sl: res = "LOSS"
                else: # SHORT
                    if l <= tp: res = "WIN"
                    elif h >= sl: res = "LOSS"
                
                if res:
                    trades_log.append({'idx': i, 'side': side, 'price': entry, 'res': res, 'tp': tp, 'sl': sl})
                    last_trade_idx = i  # Фиксируем время сделки, чтобы включить cooldown
                    break

    # --- ВИЗУАЛИЗАЦИЯ И ОТПРАВКА ---
    plt.figure(figsize=(15, 8))
    plt.plot(df['c'], color='#2c3e50', alpha=0.3, label='Цена')
    
    for t in trades_log:
        entry_color = '#3498db' if t['side'] == 'LONG' else '#e67e22'
        res_color = '#27ae60' if t['res'] == 'WIN' else '#c0392b'
        
        # Рисуем вход (треугольник) и результат (точка чуть правее)
        plt.scatter(t['idx'], t['price'], marker='^' if t['side']=='LONG' else 'v', color=entry_color, s=100, edgecolors='white')
        plt.scatter(t['idx']+1, t['price'], marker='o', color=res_color, s=40, alpha=0.8)

    plt.title(f"Smart Backtest {symbol} | Сделок: {len(trades_log)}")
    plt.grid(True, alpha=0.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    
    # Таблица для Телеграм
    table = "📋 **Результаты (последние):**\n`ID  | Тип   | Вход    | Итог`\n"
    for t in trades_log[-15:]:
        icon = "✅" if t['res'] == "WIN" else "❌"
        table += f"`{t['idx']:<4}| {t['side']:<6}| {t['price']:<8.1f}| {t['res']} {icon}`\n"

    win_count = len([t for t in trades_log if t['res']=='WIN'])
    wr = round(win_count/len(trades_log)*100, 1) if trades_log else 0
    
    caption = f"📊 **Бэктест {symbol}**\nВинрейт: **{wr}%**\nСделок: {len(trades_log)}\n\n{table}"
    
    bot.send_photo(CHAT_ID, buf, caption=caption, parse_mode="Markdown")
    plt.close()

    # --- ВИЗУАЛИЗАЦИЯ (НОВАЯ) ---
    plt.figure(figsize=(15, 8))
    plt.plot(df['c'], color='#2c3e50', alpha=0.3, label='Цена', linewidth=1)
    
    for t in trades_log:
        # Вход: Синий (Long) / Оранжевый (Short)
        entry_color = '#3498db' if t['side'] == 'LONG' else '#e67e22'
        marker = '^' if t['side'] == 'LONG' else 'v'
        plt.scatter(t['idx'], t['price'], marker=marker, color=entry_color, s=120, edgecolors='white', label=t['side'] if i==0 else "")
        
        # Результат: Зеленый (WIN) / Красный (LOSS)
        res_color = '#27ae60' if t['res'] == 'WIN' else '#c0392b'
        plt.scatter(t['idx']+2, t['price'], marker='o', color=res_color, s=50, alpha=0.7)

    plt.title(f"Детальный Бэктест {symbol} | Сделок: {len(trades_log)}")
    plt.grid(True, alpha=0.1)
    
    # Сохраняем график
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    
    # Формируем таблицу сделок (последние 15 сделок)
    table = "📋 **Последние сделки:**\n`ID  | Тип   | Вход    | Итог`\n"
    for t in trades_log[-15:]:
        emoji = "✅" if t['res'] == "WIN" else "❌"
        table += f"`{t['idx']:<4}| {t['side']:<6}| {t['price']:<8.2f}| {t['res']} {emoji}`\n"

    win_count = len([t for t in trades_log if t['res']=='WIN'])
    wr = round(win_count/len(trades_log)*100, 1) if trades_log else 0
    
    caption = f"📊 **Отчет {symbol}**\nВинрейт: **{wr}%**\nВсего сделок: {len(trades_log)}\n\n{table}"
    
    bot.send_photo(CHAT_ID, buf, caption=caption, parse_mode="Markdown")
    plt.close()

if __name__ == "__main__":
    for s in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
        run_visual_backtest(s)




