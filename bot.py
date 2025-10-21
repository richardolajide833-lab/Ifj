# super_gemini_football_bot_auto.py
# Telegram + Gemini + Football Predictions + ML + Daily Auto Message + Configurable Time

import os, json, time, threading, requests, datetime
import telebot
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from dateutil import parser as dateparser

# ========== CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
FOOTBALL_KEY   = os.getenv("FOOTBALL_KEY", "YOUR_FOOTBALL_DATA_API_KEY")
ADMIN_CHAT_ID  = os.getenv("ADMIN_CHAT_ID", "YOUR_ADMIN_CHAT_ID")  # numeric string
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": FOOTBALL_KEY}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
MODEL_FILE    = os.path.join(DATA_DIR, "rf_model.joblib")
MODEL_META    = os.path.join(DATA_DIR, "model_meta.json")
DB_MATCHES    = os.path.join(DATA_DIR, "matches.json")

DEFAULT_SETTINGS = {"hour": 9, "minute": 0}

# ========== INIT ==========
genai.configure(api_key=GEMINI_API_KEY)
GEN_MODEL = "gemini-1.5-flash"
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

def load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except:
            return default
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

settings = load_json(SETTINGS_FILE, DEFAULT_SETTINGS)

# ========== API ==========
def safe_get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=12)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def cache_matches(matches):
    cached = load_json(DB_MATCHES, [])
    existing_ids = {m.get("id") for m in cached}
    for m in matches:
        if m.get("id") not in existing_ids:
            cached.append(m)
    save_json(DB_MATCHES, cached)

# ========== FEATURES ==========
def match_to_row(m, cached=None):
    try:
        if cached is None:
            cached = load_json(DB_MATCHES, [])
        home = m["homeTeam"]["id"]
        away = m["awayTeam"]["id"]
        date_iso = m["utcDate"]
        def stats_for(team_id, before_iso):
            before = dateparser.parse(before_iso)
            recent = []
            for mm in sorted(cached, key=lambda x: x.get("utcDate", ""), reverse=True):
                try:
                    d = dateparser.parse(mm["utcDate"])
                except:
                    continue
                if d >= before: continue
                if mm.get("status") != "FINISHED": continue
                if mm.get("homeTeam", {}).get("id") == team_id or mm.get("awayTeam", {}).get("id") == team_id:
                    recent.append(mm)
                if len(recent) >= 5: break
            wins=draws=losses=gf=ga=0
            for rm in recent:
                if rm["homeTeam"]["id"] == team_id:
                    sc_h = rm["score"]["fullTime"].get("home",0)
                    sc_a = rm["score"]["fullTime"].get("away",0)
                    gf += sc_h; ga += sc_a
                    winner = rm["score"].get("winner")
                    if winner == "HOME_TEAM": wins+=1
                    elif winner == "AWAY_TEAM": losses+=1
                    else: draws+=1
                else:
                    sc_h = rm["score"]["fullTime"].get("home",0)
                    sc_a = rm["score"]["fullTime"].get("away",0)
                    gf += sc_a; ga += sc_h
                    winner = rm["score"].get("winner")
                    if winner == "AWAY_TEAM": wins+=1
                    elif winner == "HOME_TEAM": losses+=1
                    else: draws+=1
            n = max(1, len(recent))
            return {"win_rate": wins/n, "avg_gf": gf/n, "avg_ga": ga/n}
        home_stats = stats_for(home, date_iso)
        away_stats = stats_for(away, date_iso)
        winner = m.get("score", {}).get("winner")
        label = "DRAW" if winner=="DRAW" else ("HOME" if winner=="HOME_TEAM" else ("AWAY" if winner=="AWAY_TEAM" else None))
        return {
            "home_win_rate": home_stats["win_rate"],
            "home_avg_gf": home_stats["avg_gf"],
            "home_avg_ga": home_stats["avg_ga"],
            "away_win_rate": away_stats["win_rate"],
            "away_avg_gf": away_stats["avg_gf"],
            "away_avg_ga": away_stats["avg_ga"],
            "label": label
        }
    except:
        return None

# ========== ML ==========
def train_model():
    matches = load_json(DB_MATCHES, [])
    rows = [match_to_row(m, matches) for m in matches if match_to_row(m, matches)]
    df = pd.DataFrame([r for r in rows if r["label"]])
    if len(df) < 10:
        return 0
    X = df[["home_win_rate","home_avg_gf","home_avg_ga","away_win_rate","away_avg_gf","away_avg_ga"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_FILE)
    save_json(MODEL_META, {"accuracy": acc, "trained": str(datetime.datetime.utcnow())})
    return acc

def model_predict(home_id, away_id):
    if not os.path.exists(MODEL_FILE):
        return {"error":"Model not trained yet"}
    clf = joblib.load(MODEL_FILE)
    dummy = {"homeTeam":{"id":home_id},"awayTeam":{"id":away_id},"utcDate":datetime.datetime.utcnow().isoformat()}
    r = match_to_row(dummy)
    if not r: return {"error":"Insufficient data"}
    X = np.array([[r["home_win_rate"],r["home_avg_gf"],r["home_avg_ga"],r["away_win_rate"],r["away_avg_gf"],r["away_avg_ga"]]])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0].max()*100 if hasattr(clf,"predict_proba") else 50
    return {"prediction": pred, "confidence": round(prob,1)}

# ========== GEMINI ==========
def gemini_response(prompt):
    try:
        resp = genai.generate(model=GEN_MODEL, prompt=prompt, max_output_tokens=150)
        return resp.candidates[0].content if resp and resp.candidates else "No response"
    except Exception as e:
        return f"Gemini error: {e}"

# ========== TELEGRAM COMMANDS ==========
@bot.message_handler(commands=['start'])
def cmd_start(m):
    text = (
        "⚽ *Super Football Prediction Bot*\n\n"
        "Commands:\n"
        "/predict - Today's predictions\n"
        "/train - Train model\n"
        "/model_status - Show accuracy\n"
        "/settime <hour> <minute> - Change daily prediction time (admin)\n"
    )
    bot.send_message(m.chat.id, text)

@bot.message_handler(commands=['train'])
def cmd_train(m):
    if str(m.from_user.id) != str(ADMIN_CHAT_ID):
        bot.send_message(m.chat.id, "Unauthorized.")
        return
    bot.send_message(m.chat.id, "Training model... please wait ⏳")
    acc = train_model()
    if acc == 0:
        bot.send_message(m.chat.id, "Not enough data to train.")
    else:
        bot.send_message(m.chat.id, f"Model trained ✅ Accuracy: {acc:.2f}")

@bot.message_handler(commands=['model_status'])
def cmd_model_status(m):
    meta = load_json(MODEL_META, {})
    bot.send_message(m.chat.id, json.dumps(meta, indent=2))

@bot.message_handler(commands=['settime'])
def cmd_settime(m):
    if str(m.from_user.id) != str(ADMIN_CHAT_ID):
        bot.send_message(m.chat.id, "Unauthorized.")
        return
    parts = m.text.split()
    if len(parts) < 3:
        bot.send_message(m.chat.id, "Usage: /settime <hour> <minute>")
        return
    try:
        h = int(parts[1]); mi = int(parts[2])
        settings["hour"], settings["minute"] = h, mi
        save_json(SETTINGS_FILE, settings)
        bot.send_message(m.chat.id, f"✅ Time updated to {h:02d}:{mi:02d}")
    except:
        bot.send_message(m.chat.id, "Invalid time format")

@bot.message_handler(commands=['predict'])
def cmd_predict(m):
    msg = predict_today()
    bot.send_message(m.chat.id, msg)

# ========== DAILY TASK ==========
def predict_today():
    today = datetime.date.today().isoformat()
    res = safe_get(f"{BASE_URL}/matches", params={"dateFrom":today,"dateTo":today})
    matches = res.get("matches", [])
    if not matches:
        return f"No matches today ({today})."
    out = f"*Predictions for {today}:*\n\n"
    for match in matches:
        home = match["homeTeam"]["id"]; away = match["awayTeam"]["id"]
        pred = model_predict(home, away)
        if "error" in pred: continue
        out += f"{match['homeTeam']['name']} vs {match['awayTeam']['name']} → *{pred['prediction']}* ({pred['confidence']}%)\n"
    try:
        advice = gemini_response(f"Give short football betting insight for {today}.")
        out += f"\n📊 Gemini Insight:\n{advice}"
    except:
        pass
    return out

def daily_alert():
    last_sent_date = None
    while True:
        now = datetime.datetime.now()
        if (now.hour == settings["hour"] and now.minute == settings["minute"]):
            if last_sent_date != now.date():
                msg = predict_today()
                bot.send_message(ADMIN_CHAT_ID, msg)
                last_sent_date = now.date()
                time.sleep(60)
        time.sleep(5)

# ========== START ==========
if __name__ == "__main__":
    threading.Thread(target=daily_alert, daemon=True).start()
    print("Bot started ✅")
    bot.infinity_polling()
