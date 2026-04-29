"""
============================================================
 AI-Pacer Day 2 통합 버전
============================================================
 추가 모듈 (Day 1 → Day 2 변경사항):
  ① 환경 위험 평가  : WBGT 기반 고온/저온/바람 코칭
                      (Gabbett 2016, ACSM Cold-Weather Consensus 2021,
                       Beaumont & Polidori 2025 - wind metabolic cost)
  ② ACWR 부상 추적  : Acute:Chronic Workload Ratio (EWMA 방식)
                      (Gabbett 2016 BJSM, Williams 2017, Wang 2024)
  ③ 온보딩 설문     : Cold Start 해결 - ACSM Pre-participation
  ④ 페이스 윈도우   : 60초 / 5분 윈도우 분석
                      → 갑작스런 하락 vs 점진적 하락 구분
  ⑤ STT 부상 확인   : Web Speech Recognition API
                      → 부상 키워드 감지 시 즉시 정지 권고
  ⑥ 구간별 멘트     : 0-30% / 30-70% / 70-90% / 90-100% 톤 변화
  ⑦ 우선순위 시스템 : 응급 > 부상 > 환경 > 페이스 > 동기부여
                      카테고리별 TTS 쿨다운 분리 운영
============================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import folium
import json
import math
import pandas as pd
from streamlit_folium import st_folium
from openai import OpenAI
from datetime import datetime, timedelta

# ===== API 키 =====
OPENAI_API_KEY      = "key"
OPENWEATHER_API_KEY = "key"
KAKAO_REST_API_KEY  = "key"
TMAP_API_KEY        = "key"

client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="AI-Pacer", layout="wide")

# ============================================================
# Session State 초기화
# ============================================================
defaults = {
    # 기존
    "running": False, "finished": False,
    "gps_track": [], "pace_history": [],
    "start_time": None, "route_data": None,
    "s_lat": None, "s_lng": None,
    "e_lat": None, "e_lng": None,
    "s_name": "", "e_name": "",
    "current_lat": None, "current_lng": None,
    "current_pace": 0.0, "target_pace": 6.0,
    "current_nav_step": 0,
    # Day 2 추가
    "profile_set": False,
    "baseline_weekly_km": 0.0,
    "user_level": "입문",
    "recent_runs": [],     # ACWR용 일별 거리 누적: [(date, km), ...]
    "env_risk": None,      # 현재 환경 위험 등급
    "env_messages": [],    # 환경 코칭 메시지
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# 1. 환경 분석 모듈 (WBGT)
#    근거: Gabbett 2016, ACSM Cold-Weather Consensus 2021,
#          Beaumont & Polidori 2025
# ============================================================
def calculate_wbgt(temp_c: float, humidity: float) -> float:
    """
    Australian Bureau of Meteorology 근사식
    WBGT ≈ 0.567*Td + 0.393*e + 3.94
    e = 수증기압 (hPa)
    실외 직사광 미반영 단순 추정치
    """
    e = (humidity / 100.0) * 6.105 * math.exp(
        17.27 * temp_c / (237.7 + temp_c))
    return 0.567 * temp_c + 0.393 * e + 3.94


def get_environment_risk(temp_c: float, humidity: float,
                         wind_ms: float) -> tuple:
    """
    환경 위험도 평가 → (등급, 메시지 리스트, WBGT)
    - 고온: WBGT 기반 (ACSM 가이드)
    - 저온: 근육 온도 32°C 이하 위험 (Cold Muscles 연구)
    - 바람: 5m/s 이상부터 유의미 영향 (Beaumont 2025)
    """
    wbgt = calculate_wbgt(temp_c, humidity)
    risk = "normal"
    msgs = []

    # 고온 평가
    if wbgt >= 30:
        risk = "extreme"
        msgs.append(f"🚨 폭염 위험 (WBGT {wbgt:.1f}°C). 러닝 중단 권고.")
    elif wbgt >= 28:
        risk = "high"
        msgs.append(f"⚠️ 고온 경고 (WBGT {wbgt:.1f}°C). "
                    f"페이스 10초/km 늦추고 5분마다 수분 보충하세요.")
    elif wbgt >= 23:
        risk = "moderate"
        msgs.append(f"☀️ 더위 주의 (WBGT {wbgt:.1f}°C). "
                    f"페이스 5초/km 여유, 수분 자주 챙기세요.")

    # 저온 평가
    if temp_c <= -9:
        risk = "high"
        msgs.append(f"❄️ 한랭 경고 ({temp_c}°C). "
                    f"고강도 운동 비권장. 워밍업 15분 이상 필수.")
    elif temp_c <= 0:
        if risk == "normal":
            risk = "moderate"
        msgs.append(f"❄️ 영하 환경 ({temp_c}°C). "
                    f"워밍업 10분, 첫 1km 페이스 30초 늦추세요.")
    elif temp_c <= 10:
        msgs.append(f"🧥 쌀쌀한 날씨 ({temp_c}°C). 워밍업 충분히.")

    # 바람 평가 (Beaumont & Polidori 2025: 30km/h 역풍 = +37% 에너지)
    if wind_ms >= 5.5:  # ~20km/h
        msgs.append(f"💨 강풍 ({wind_ms:.1f}m/s). "
                    f"역풍 구간 페이스 10초/km 손해 예상. 초반 여유.")
    elif wind_ms >= 4.2:  # ~15km/h
        msgs.append(f"💨 바람 강함 ({wind_ms:.1f}m/s). 페이스 5초/km 여유.")

    if not msgs:
        msgs.append(f"✅ 쾌적한 환경 (WBGT {wbgt:.1f}°C, "
                    f"바람 {wind_ms:.1f}m/s).")
    return risk, msgs, wbgt


# ============================================================
# 2. ACWR 모듈 (EWMA 방식)
#    근거: Gabbett 2016, Williams et al. 2017
# ============================================================
def calculate_acwr_ewma(daily_loads: list,
                        baseline: float = 0.0,
                        days_since_signup: int = 0) -> tuple:
    """
    EWMA 방식 ACWR 계산
      λ_acute  = 2/(7+1)  ≈ 0.25
      λ_chronic= 2/(28+1) ≈ 0.069

    Cold Start 처리:
      - 가입 후 28일 미만: baseline(자가보고)을 가중 혼합
      - Progressive weighting (4주 점진적 전환)

    Returns: (acwr, ewma_acute, ewma_chronic, confidence)
      confidence: 'self_report' | 'mixed' | 'measured'
    """
    LAMBDA_ACUTE = 0.25
    LAMBDA_CHRONIC = 0.069

    # baseline: 주간 km → 일평균 km
    baseline_daily = baseline / 7.0 if baseline > 0 else 0.0

    if not daily_loads and baseline_daily == 0:
        return None, 0, 0, "no_data"

    # 28일 미만 → 자가보고 가상 데이터로 채움
    if days_since_signup < 28:
        weight = days_since_signup / 28.0  # 0 → 1
        confidence = "self_report" if days_since_signup < 7 else "mixed"
    else:
        weight = 1.0
        confidence = "measured"

    # 초기값 = baseline_daily (없으면 첫 데이터)
    init = baseline_daily if baseline_daily > 0 else (
        daily_loads[0] if daily_loads else 0)
    ewma_a = init
    ewma_c = init

    for load in daily_loads:
        # 측정 데이터에 weight, baseline에 (1-weight) 혼합
        effective = load * weight + baseline_daily * (1 - weight)
        ewma_a = effective * LAMBDA_ACUTE + ewma_a * (1 - LAMBDA_ACUTE)
        ewma_c = effective * LAMBDA_CHRONIC + ewma_c * (1 - LAMBDA_CHRONIC)

    if ewma_c < 0.01:
        return None, ewma_a, ewma_c, confidence

    acwr = ewma_a / ewma_c
    return acwr, ewma_a, ewma_c, confidence


def estimate_baseline_load(freq_label: str, dist_label: str) -> float:
    """온보딩 설문 → 추정 주간 거리(km)"""
    freq_map = {"안 뜀": 0, "1-2회": 1.5, "3-4회": 3.5, "5회 이상": 5.5}
    dist_map = {"3km 이하": 2.5, "3-5km": 4.0,
                "5-10km": 7.0, "10km 이상": 12.0}
    return freq_map.get(freq_label, 0) * dist_map.get(dist_label, 0)


def get_acwr_advice(acwr, confidence: str) -> tuple:
    """ACWR → (라벨, 권고메시지, status)"""
    if acwr is None:
        return ("데이터 부족", "온보딩 설문을 완료해주세요.", "info")

    note = ""
    if confidence == "self_report":
        note = " (자가보고 기반 추정)"
    elif confidence == "mixed":
        note = " (자가보고 + 실측 혼합)"

    if acwr < 0.8:
        return (f"⚠️ 저부하 ACWR {acwr:.2f}{note}",
                "오랜만에 뛰는 거라면 첫 1~2km 천천히 시작하세요.", "warning")
    elif acwr <= 1.3:
        return (f"✅ Sweet Spot ACWR {acwr:.2f}{note}",
                "적정 훈련량입니다. 평소대로 진행하세요.", "success")
    elif acwr <= 1.5:
        return (f"⚠️ 주의 구간 ACWR {acwr:.2f}{note}",
                "훈련량 빠르게 증가 중. 회복 챙기세요.", "warning")
    else:
        return (f"🚨 위험 구간 ACWR {acwr:.2f}{note}",
                "부상 위험 높음. 오늘은 회복 페이스 권장.", "error")


# ============================================================
# 기존 API 함수 (Day 1 유지)
# ============================================================
def get_kakao_coords(address):
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    try:
        res = requests.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            headers=headers, params={"query": address}, timeout=5)
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            if docs:
                d = docs[0]
                return float(d['y']), float(d['x']), d.get('address_name', address)
        res2 = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers=headers, params={"query": address, "size": 1}, timeout=5)
        if res2.status_code == 200:
            docs2 = res2.json().get("documents", [])
            if docs2:
                d = docs2[0]
                return float(d['y']), float(d['x']), d.get('place_name', address)
            return None, None, f"'{address}' 검색 결과 없음"
        return None, None, f"카카오 오류 {res2.status_code}"
    except Exception as e:
        return None, None, str(e)


def get_tmap_walking_route(s_lat, s_lng, e_lat, e_lng,
                            s_name="출발지", e_name="목적지"):
    url = (f"https://apis.openapi.sk.com/tmap/routes/pedestrian"
           f"?version=1&format=json&appKey={TMAP_API_KEY}")
    body = {
        "startX": str(s_lng), "startY": str(s_lat),
        "endX":   str(e_lng), "endY":   str(e_lat),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO",
        "startName": s_name, "endName": e_name, "searchOption": "0",
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          json=body, timeout=15)
        if r.status_code != 200:
            return None, f"Tmap API 오류 {r.status_code}"
        features = r.json().get("features", [])
        if not features:
            return None, "경로 결과 없음"
        path, steps = [], []
        for f in features:
            if f["geometry"]["type"] == "LineString":
                for c in f["geometry"]["coordinates"]:
                    path.append([c[1], c[0]])
        for f in features:
            if f["geometry"]["type"] == "Point":
                props = f["properties"]
                desc = props.get("description", "")
                dist = props.get("distance", 0)
                pt = props.get("pointType", "")
                coord = f["geometry"]["coordinates"]
                if pt not in ("SP",) and desc:
                    steps.append({
                        "description": desc, "distance": float(dist),
                        "pointType": pt,
                        "lat": coord[1], "lng": coord[0],
                    })
        summary = features[0]["properties"]
        return {
            "path": path, "steps": steps,
            "dist_m": summary.get("totalDistance", 0),
            "time_sec": summary.get("totalTime", 0),
        }, None
    except Exception as e:
        return None, str(e)


def get_weather_extended(lat, lng):
    """Day 1: temp/hum/desc → Day 2: + 풍속/풍향 추가"""
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lng}&appid={OPENWEATHER_API_KEY}"
               f"&units=metric&lang=kr")
        res = requests.get(url, timeout=5).json()
        return {
            "temp":     res['main']['temp'],
            "humidity": res['main']['humidity'],
            "desc":     res['weather'][0]['description'],
            "wind_ms":  res.get('wind', {}).get('speed', 0),
            "wind_deg": res.get('wind', {}).get('deg', 0),
        }
    except Exception:
        return {"temp": 20.0, "humidity": 50, "desc": "알 수 없음",
                "wind_ms": 0, "wind_deg": 0}


def calc_distance_m(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# 3. 자동 GPS + 페이스 윈도우 + STT 부상확인 + 구간별 멘트
#    Day 2 강화 컴포넌트 (JS)
# ============================================================
def build_auto_component(steps, target_pace, total_dist_m, env_messages):
    steps_json = json.dumps(steps, ensure_ascii=False)
    env_msgs_json = json.dumps(env_messages, ensure_ascii=False)
    return f"""
<div style="background:#1a1a2e; border-radius:10px; padding:12px;
            color:#e0e0e0; font-family:monospace; font-size:13px;
            border:1px solid #333;">
  <div id="gps-info">📡 GPS 초기화 중...</div>
  <div id="nav-info" style="margin-top:8px; font-size:15px;
       font-weight:bold; color:#64b5f6;">🧭 내비게이션 대기 중...</div>
  <div id="pace-info" style="margin-top:6px; color:#a5d6a7;">
       ⚡ 페이스 측정 중...</div>
  <div id="zone-info" style="margin-top:6px; color:#ffb74d;">
       📍 진행률 계산 중...</div>
  <div id="pain-check" style="margin-top:8px; padding:8px;
       background:#3e2723; border-radius:6px; display:none;
       color:#ffccbc; font-weight:bold;"></div>
  <div id="speech-log" style="margin-top:6px; font-size:11px;
       color:#888;"></div>
</div>

<script>
// ========== 설정 ==========
const STEPS         = {steps_json};
const TARGET_PACE   = {target_pace};
const TOTAL_DIST    = {total_dist_m};
const ENV_MESSAGES  = {env_msgs_json};

const WAYPOINT_RADIUS = 30;
const SUDDEN_DROP     = 0.08;   // 8% 갑작스런 하락
const GRADUAL_DROP    = 0.05;   // 5% 점진적 하락

// 카테고리별 TTS 쿨다운 (우선순위별 분리)
const COOLDOWN = {{
  emergency : 0,        // 즉시
  injury    : 5000,
  pace      : 30000,
  motivation: 60000,
  env       : 120000,
}};

// ========== 상태 ==========
let currentStep   = 0;
let lastNavSpeak  = 0;
let lastByCat     = {{ pace:0, motivation:0, injury:0, env:0 }};
let painCheckActive = false;
let painCheckStart  = 0;

// 페이스 윈도우 (시계열)
let paceLog = []; // [{{ts, pace}}, ...]
let distLog = []; // [{{ts, dist}}, ...]
let totalDist = 0;
let lastLat = null, lastLng = null;
let lastTs  = null;

// 부상 키워드 사전
const KEYWORDS = {{
  safe:      ['괜찮', '문제없', '오케이', 'ok', '좋아'],
  injury:    ['무릎', '발목', '종아리', '햄스트링', '허벅지',
              '발바닥', '허리', '아파', '시려', '땡겨', '저려', '쥐'],
  emergency: ['가슴', '숨', '어지', '토할', '쓰러'],
  fatigue:   ['힘들', '지쳐', '못하겠']
}};

// ========== TTS ==========
function speak(text, category) {{
  if (!window.speechSynthesis) return;
  const now = Date.now();
  if (category && lastByCat[category] !== undefined) {{
    if (now - lastByCat[category] < COOLDOWN[category]) return;
    lastByCat[category] = now;
  }}
  // 응급/부상은 기존 발화 무시하고 우선
  if (category === 'emergency' || category === 'injury') {{
    window.speechSynthesis.cancel();
  }}
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'ko-KR'; u.rate = 1.0; u.pitch = 1.0;
  window.speechSynthesis.speak(u);
  document.getElementById('speech-log').innerText = '🔊 ' + text;
}}

function haversine(lat1, lng1, lat2, lng2) {{
  const R = 6371000;
  const dLat = (lat2-lat1)*Math.PI/180;
  const dLng = (lng2-lng1)*Math.PI/180;
  const a = Math.sin(dLat/2)**2
          + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)
          * Math.sin(dLng/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

// ========== 페이스 윈도우 분석 ==========
function getWindowAvg(seconds) {{
  const cutoff = Date.now() - seconds*1000;
  const recent = paceLog.filter(p => p.ts >= cutoff && p.pace > 0);
  if (recent.length === 0) return 0;
  return recent.reduce((s,p) => s+p.pace, 0) / recent.length;
}}

function detectPacePattern() {{
  const w60s  = getWindowAvg(60);
  const w300s = getWindowAvg(300);
  if (w60s === 0 || w300s === 0) return 'init';

  const sudden = (w60s - w300s) / w300s;
  // 갑작스런 하락 (페이스 ↑ = 느려짐)
  if (sudden > SUDDEN_DROP)  return 'sudden_drop';
  if (sudden < -SUDDEN_DROP) return 'sudden_spike';

  // 점진적 하락: 5분 평균이 베이스 대비 느려짐 + 단조 증가 추세
  if (paceLog.length > 20) {{
    const baseline = paceLog.slice(0, 10)
      .filter(p => p.pace > 0).reduce((s,p) => s+p.pace, 0) /
      Math.max(1, paceLog.slice(0,10).filter(p=>p.pace>0).length);
    if (baseline > 0 && (w300s - baseline)/baseline > GRADUAL_DROP) {{
      return 'gradual_drop';
    }}
  }}
  return 'stable';
}}

// ========== STT 부상 확인 ==========
let recognition = null;
function startPainCheck() {{
  if (painCheckActive) return;
  painCheckActive = true;
  painCheckStart = Date.now();

  const box = document.getElementById('pain-check');
  box.style.display = 'block';
  box.innerText = '🎙️ 응답 대기 중... "괜찮아" 또는 아픈 부위 말해주세요.';

  speak('페이스가 떨어졌어요. 어디 불편한 곳 있나요? ' +
        '괜찮으면 괜찮아라고 답해주세요.', 'injury');

  if (!('webkitSpeechRecognition' in window)
      && !('SpeechRecognition' in window)) {{
    box.innerText = '⚠️ 이 브라우저는 음성인식 미지원. ' +
                    '버튼으로 응답하세요.';
    return;
  }}
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'ko-KR';
  recognition.interimResults = false;

  recognition.onresult = (e) => {{
    const text = e.results[0][0].transcript.toLowerCase();
    box.innerText = '👂 인식: "' + text + '"';
    handlePainResponse(text);
  }};
  recognition.onerror = () => {{
    box.innerText = '⚠️ 음성 인식 오류. 페이스 모니터링 계속.';
    painCheckActive = false;
  }};
  recognition.onend = () => {{
    if (painCheckActive && Date.now() - painCheckStart > 15000) {{
      box.innerText = '⏱️ 응답 없음. 안전을 위해 페이스 줄여주세요.';
      speak('응답이 없어요. 안전하게 페이스 줄이고 걸어주세요.',
            'injury');
      painCheckActive = false;
    }}
  }};
  try {{ recognition.start(); }} catch(e) {{}}
}}

function handlePainResponse(text) {{
  const box = document.getElementById('pain-check');

  // 응급 우선 체크
  for (const k of KEYWORDS.emergency) {{
    if (text.includes(k)) {{
      box.style.background = '#b71c1c';
      box.innerText = '🚨 응급 키워드 감지: ' + k;
      speak('지금 즉시 멈추고 안전한 곳에 앉으세요. ' +
            '증상 안 가라앉으면 119에 연락하세요.', 'emergency');
      painCheckActive = false;
      return;
    }}
  }}
  // 부상 키워드
  for (const k of KEYWORDS.injury) {{
    if (text.includes(k)) {{
      box.style.background = '#bf360c';
      box.innerText = '⚠️ 부상 의심 키워드: ' + k;
      speak('지금 바로 멈추세요. 통증을 무시하면 ' +
            '몇 주 못 뛰는 부상으로 이어져요. 걸으면서 ' +
            '통증이 가라앉는지 확인하세요.', 'injury');
      painCheckActive = false;
      return;
    }}
  }}
  // 안전 응답
  for (const k of KEYWORDS.safe) {{
    if (text.includes(k)) {{
      box.style.background = '#1b5e20';
      box.innerText = '✅ 안전 응답 확인';
      speak('다행이에요. 호흡 가다듬고 천천히 회복해봐요.',
            'pace');
      painCheckActive = false;
      return;
    }}
  }}
  // 피로 (정지는 X, 페이스 조절)
  for (const k of KEYWORDS.fatigue) {{
    if (text.includes(k)) {{
      box.innerText = '😮‍💨 피로 신호';
      speak('힘드시군요. 페이스 1~2초 늦추고 ' +
            '4초 들이마시고 4초 내쉬는 호흡으로.', 'pace');
      painCheckActive = false;
      return;
    }}
  }}
  box.innerText = '❓ 키워드 미감지: "' + text + '"';
  painCheckActive = false;
}}

// ========== 구간별 동기부여 ==========
function getZoneMessage(progress, paceStatus) {{
  if (progress < 0.3) {{
    return paceStatus === 'fast'
      ? '초반에 너무 빨라요. 후반 위해 페이스 낮추세요.'
      : '좋은 시작이에요. 호흡 안정시키며 가요.';
  }} else if (progress < 0.7) {{
    return paceStatus === 'slow'
      ? '중반 구간이에요. 자세 점검하고 어깨 힘 빼세요.'
      : '리듬 좋아요. 이 페이스 유지하면 됩니다.';
  }} else if (progress < 0.9) {{
    return '이제 끝이 보여요. 지금 무너지면 아깝잖아요. 버텨봐요.';
  }} else {{
    return '마지막 구간! 다 왔어요. 끝까지 갑시다!';
  }}
}}

// ========== Streamlit 데이터 전송 ==========
function sendToStreamlit(lat, lng, pace, step, dist) {{
  try {{
    window.parent.postMessage({{
      type: 'streamlit:setComponentValue',
      value: JSON.stringify({{lat, lng, pace, step, dist,
                              ts: Date.now()}})
    }}, '*');
  }} catch(e) {{}}
}}

// ========== GPS 콜백 ==========
function onGPS(pos) {{
  const lat = pos.coords.latitude;
  const lng = pos.coords.longitude;
  const spd = pos.coords.speed;
  const now = Date.now();

  // 거리 누적
  if (lastLat !== null) {{
    const seg = haversine(lastLat, lastLng, lat, lng);
    if (seg < 50) totalDist += seg;
  }}
  lastLat = lat; lastLng = lng; lastTs = now;

  // 페이스
  let pace = 0;
  if (spd && spd > 0.3) pace = (1000 / spd) / 60;
  if (pace > 0) paceLog.push({{ts: now, pace}});
  if (paceLog.length > 600) paceLog.shift();

  const paceStr = pace > 0
    ? Math.floor(pace) + '분 ' + Math.round((pace%1)*60) + '초/km'
    : '측정중';
  document.getElementById('gps-info').innerText =
    '✅ GPS | ' + lat.toFixed(5) + ', ' + lng.toFixed(5)
    + ' | ' + paceStr + ' | ' + Math.round(totalDist) + 'm';

  // 진행률
  const progress = TOTAL_DIST > 0 ? Math.min(1, totalDist/TOTAL_DIST) : 0;
  document.getElementById('zone-info').innerText =
    '📍 진행률 ' + (progress*100).toFixed(1) + '% / ' +
    Math.round(TOTAL_DIST) + 'm 중 ' + Math.round(totalDist) + 'm';

  // ── 턴바이턴 ──
  if (currentStep < STEPS.length) {{
    const step = STEPS[currentStep];
    const dist = haversine(lat, lng, step.lat, step.lng);
    document.getElementById('nav-info').innerText =
      '🧭 [' + (currentStep+1) + '/' + STEPS.length + '] '
      + (step.distance > 0 ? Math.round(step.distance) + 'm 앞  ' : '')
      + step.description;
    if (dist < WAYPOINT_RADIUS && now - lastNavSpeak > 5000) {{
      lastNavSpeak = now;
      speak(step.description, null);  // 내비는 항상 발화
      if (currentStep + 1 < STEPS.length) {{
        currentStep++;
      }} else {{
        speak('목적지에 도착했습니다. 수고하셨습니다!', null);
      }}
    }}
  }}

  // ── 페이스 패턴 분석 (응답대기 중엔 패스) ──
  if (!painCheckActive && pace > 0) {{
    const pattern = detectPacePattern();
    let paceCategory = 'stable';

    if (pattern === 'sudden_drop') {{
      // 갑작스런 하락 → STT 부상 확인 트리거
      startPainCheck();
      paceCategory = 'slow';
    }} else if (pattern === 'gradual_drop') {{
      const distKm = totalDist / 1000;
      let msg = '페이스가 조금씩 떨어지고 있어요. ';
      if (distKm < 5) {{
        msg += '초반이 너무 빨랐을 수 있어요. 호흡 가다듬으세요.';
      }} else if (distKm < 15) {{
        msg += '자세 점검하세요. 어깨 힘 빼고 시선 앞으로.';
      }} else {{
        msg += '에너지 떨어지는 신호예요. 젤이나 당분 챙기고 ' +
               '4초 호흡 리듬으로 바꾸세요.';
      }}
      speak(msg, 'pace');
      paceCategory = 'slow';
    }} else if (pattern === 'sudden_spike') {{
      speak('페이스 갑자기 빨라졌어요. 의도한 거 아니면 ' +
            '원래 페이스로 돌리세요.', 'pace');
      paceCategory = 'fast';
    }} else {{
      // 단순 ±10% 임계 (보조)
      const diff = (pace - TARGET_PACE) / TARGET_PACE;
      if (diff > 0.10) {{
        speak('목표 페이스보다 느려요. 조금씩 올려보세요.', 'pace');
        paceCategory = 'slow';
      }} else if (diff < -0.10) {{
        speak('너무 빨라요. 후반 위해 속도 줄이세요.', 'pace');
        paceCategory = 'fast';
      }}
    }}

    // 페이스 표시 색상
    const info = document.getElementById('pace-info');
    if (paceCategory === 'slow') {{
      info.style.color = '#ef9a9a';
      info.innerText = '🐢 페이스 느림 — ' + paceStr;
    }} else if (paceCategory === 'fast') {{
      info.style.color = '#fff176';
      info.innerText = '🚀 페이스 빠름 — ' + paceStr;
    }} else {{
      info.style.color = '#a5d6a7';
      info.innerText = '✅ 페이스 정상 — ' + paceStr;
    }}

    // ── 구간별 동기부여 (낮은 우선순위) ──
    if (now - lastByCat.motivation > COOLDOWN.motivation) {{
      const zoneMsg = getZoneMessage(progress, paceCategory);
      speak(zoneMsg, 'motivation');
    }}
  }}

  sendToStreamlit(lat, lng, pace, currentStep, totalDist);
}}

// ========== 시작 시퀀스 ==========
if (navigator.geolocation) {{
  // 환경 경고 먼저 발화
  setTimeout(() => speak('AI 페이서 시작합니다.', null), 500);
  ENV_MESSAGES.forEach((m, i) => {{
    setTimeout(() => speak(m, 'env'),
               2000 + i * 4500);
  }});
  if (STEPS.length > 0) {{
    setTimeout(() => speak('첫 안내. ' + STEPS[0].description, null),
               2000 + ENV_MESSAGES.length * 4500);
  }}
  navigator.geolocation.watchPosition(onGPS,
    err => {{
      document.getElementById('gps-info').innerText =
        '❌ GPS 오류: ' + err.message;
    }},
    {{enableHighAccuracy: true, timeout: 5000, maximumAge: 0}}
  );
}} else {{
  document.getElementById('gps-info').innerText = '❌ GPS 미지원';
}}
</script>
"""


# ============================================================
# 사이드바: 온보딩 + ACWR 대시보드
# ============================================================
with st.sidebar:
    st.header("👤 사용자 프로필")

    if not st.session_state.profile_set:
        st.warning("초기 프로필을 설정해주세요 (ACWR Cold Start 해결)")

    with st.expander("🎯 온보딩 설문",
                      expanded=not st.session_state.profile_set):
        st.caption("💡 ACSM Pre-participation 가이드 기반")
        freq = st.selectbox("최근 한 달 주간 러닝 빈도",
                             ["안 뜀", "1-2회", "3-4회", "5회 이상"])
        dist = st.selectbox("1회당 평균 거리",
                             ["3km 이하", "3-5km",
                              "5-10km", "10km 이상"])
        level = st.radio("러닝 레벨",
                          ["입문 (3개월 미만)", "초급 (6개월)",
                           "중급 (1년+)", "상급 (마라톤 경험)"])
        if st.button("프로필 저장", use_container_width=True):
            baseline = estimate_baseline_load(freq, dist)
            st.session_state.baseline_weekly_km = baseline
            st.session_state.user_level = level
            st.session_state.profile_set = True
            st.success(f"✅ 추정 주간 거리 {baseline:.1f}km")
            st.rerun()

    st.markdown("---")
    st.subheader("📊 ACWR 부상위험 추적")

    # 최근 28일 합 vs 추정
    today = datetime.now().date()
    daily = []
    if st.session_state.recent_runs:
        # 일자별 합산
        df_run = pd.DataFrame(
            st.session_state.recent_runs, columns=["date", "km"])
        df_run["date"] = pd.to_datetime(df_run["date"]).dt.date
        agg = df_run.groupby("date")["km"].sum().to_dict()
        for i in range(28, -1, -1):
            d = today - timedelta(days=i)
            daily.append(agg.get(d, 0.0))

    # 가입 후 일수 (간이: 첫 기록부터)
    days_since = 0
    if st.session_state.recent_runs:
        first = pd.to_datetime(
            st.session_state.recent_runs[0][0]).date()
        days_since = (today - first).days

    acwr, ewma_a, ewma_c, conf = calculate_acwr_ewma(
        daily, st.session_state.baseline_weekly_km, days_since)
    label, advice, status = get_acwr_advice(acwr, conf)

    if status == "success":
        st.success(label)
    elif status == "warning":
        st.warning(label)
    elif status == "error":
        st.error(label)
    else:
        st.info(label)
    st.caption(advice)

    if acwr is not None:
        col_x, col_y = st.columns(2)
        col_x.metric("Acute (7d 평균)", f"{ewma_a*7:.1f} km")
        col_y.metric("Chronic (28d 평균)", f"{ewma_c*7:.1f} km")
        # Sweet Spot 시각화
        sweet_pos = min(max(acwr, 0.5), 2.0)
        st.caption(f"📍 Sweet Spot: 0.8 ≤ ACWR ≤ 1.3 (Gabbett 2016)")


# ============================================================
# 메인 UI
# ============================================================
st.title("🏃‍♂️ AI-Pacer (Day 2)")
tab_setup, tab_running, tab_result = st.tabs(
    ["⚙️ 경로 설정", "🏃 러닝 중", "📊 결과 기록"])

# ============================================================
# TAB 1: 경로 설정 + 환경 분석
# ============================================================
with tab_setup:
    st.subheader("📍 경로 및 목표 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_input = st.text_input("출발지", placeholder="예: 서강대학교")
        target_pace = st.slider("목표 페이스 (min/km)", 2.0, 12.0,
                                 st.session_state.target_pace, step=0.1)
        st.session_state.target_pace = target_pace
        pm = int(target_pace)
        ps = int((target_pace % 1) * 60)
        st.caption(f"🎯 1km를 {pm}분 {ps:02d}초에 주파")
    with col2:
        end_input = st.text_input("목적지", placeholder="예: 마포구청")

    if st.button("🔍 경로 탐색", type="primary", use_container_width=True):
        if not start_input or not end_input:
            st.error("출발지와 목적지를 입력해주세요.")
        else:
            with st.spinner("좌표 검색 중..."):
                s_lat, s_lng, s_name = get_kakao_coords(start_input)
                e_lat, e_lng, e_name = get_kakao_coords(end_input)
            if s_lat is None:
                st.error(f"출발지 오류: {s_name}")
            elif e_lat is None:
                st.error(f"목적지 오류: {e_name}")
            else:
                with st.spinner("Tmap 도보 경로 계산 중..."):
                    route, err = get_tmap_walking_route(
                        s_lat, s_lng, e_lat, e_lng, s_name, e_name)
                if err:
                    st.error(f"경로 오류: {err}")
                else:
                    st.session_state.update({
                        "s_lat": s_lat, "s_lng": s_lng, "s_name": s_name,
                        "e_lat": e_lat, "e_lng": e_lng, "e_name": e_name,
                        "route_data": route,
                    })
                    st.success(f"✅ {s_name} → {e_name}")

    if st.session_state.route_data:
        rd = st.session_state.route_data
        dist_km = rd["dist_m"] / 1000
        est_min = dist_km * st.session_state.target_pace

        # ── 환경 위험 분석 ──
        weather = get_weather_extended(
            st.session_state.s_lat, st.session_state.s_lng)
        risk, env_msgs, wbgt = get_environment_risk(
            weather["temp"], weather["humidity"], weather["wind_ms"])

        # 세션에 저장 (러닝 중 컴포넌트로 전달)
        st.session_state.env_risk = risk
        st.session_state.env_messages = env_msgs

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("총 거리", f"{dist_km:.2f} km")
        col_b.metric("예상 완주",
                     f"{int(est_min)}분 {int((est_min%1)*60)}초")
        col_c.metric("기온/습도",
                     f"{weather['temp']:.0f}°C / {weather['humidity']}%",
                     delta=weather['desc'], delta_color="off")
        col_d.metric("WBGT",
                     f"{wbgt:.1f}°C",
                     delta=f"바람 {weather['wind_ms']:.1f}m/s",
                     delta_color="off")

        # 환경 위험 알림
        st.subheader("🌤️ 환경 분석 (출발 전 브리핑)")
        if risk == "extreme":
            for m in env_msgs:
                st.error(m)
        elif risk == "high":
            for m in env_msgs:
                st.warning(m)
        elif risk == "moderate":
            for m in env_msgs:
                st.info(m)
        else:
            for m in env_msgs:
                st.success(m)
        st.caption("📚 근거: ACSM Heat/Cold Guidelines, "
                   "Beaumont & Polidori (2025) wind metabolic cost study")

        # 지도
        path = rd["path"]
        m = folium.Map(
            location=[(st.session_state.s_lat+st.session_state.e_lat)/2,
                       (st.session_state.s_lng+st.session_state.e_lng)/2],
            zoom_start=15)
        folium.PolyLine(path, color="#1E88E5",
                         weight=6, opacity=0.9).add_to(m)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                      tooltip=f"출발: {st.session_state.s_name}",
                      icon=folium.Icon(color="green",
                                        icon="play")).add_to(m)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                      tooltip=f"도착: {st.session_state.e_name}",
                      icon=folium.Icon(color="red",
                                        icon="flag")).add_to(m)
        st_folium(m, width=None, height=420)

        with st.expander("🧭 전체 경로 안내 보기"):
            for i, s in enumerate(rd["steps"]):
                dist_str = f"  ({int(s['distance'])}m)" \
                           if s['distance'] else ""
                st.write(f"**{i+1}.** {s['description']}{dist_str}")

        # ACWR 위험 시 경고
        if not st.session_state.profile_set:
            st.warning("⚠️ 사이드바에서 온보딩 설문 먼저 완료해주세요.")

        if st.button("🏃 러닝 시작!", type="primary",
                      use_container_width=True):
            st.session_state.update({
                "running": True, "finished": False,
                "gps_track": [], "pace_history": [],
                "start_time": datetime.now(),
                "current_nav_step": 0,
            })
            st.success("러닝 시작! '러닝 중' 탭으로 이동하세요.")


# ============================================================
# TAB 2: 러닝 중
# ============================================================
with tab_running:
    if not st.session_state.running and not st.session_state.finished:
        st.info("경로 설정 탭에서 러닝을 시작하세요.")
        st.stop()
    if st.session_state.finished:
        st.success("러닝 종료. 결과 기록 탭을 확인하세요.")
        st.stop()

    rd = st.session_state.route_data
    steps = rd["steps"] if rd else []
    total_m = rd["dist_m"] if rd else 0

    st.subheader("📡 실시간 자동 안내 + 부상 모니터링")
    st.caption("🔊 폰 음소거 해제 + 마이크 권한 허용 필수")
    components.html(
        build_auto_component(
            steps, st.session_state.target_pace,
            total_m, st.session_state.env_messages or []),
        height=210)

    st.markdown("---")
    st.subheader("⚡ 페이스 대시보드")
    target = st.session_state.target_pace
    current = st.session_state.current_pace or target
    pace_diff = current - target
    elapsed = int((datetime.now() - st.session_state.start_time)
                  .total_seconds()) if st.session_state.start_time else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 목표", f"{target:.1f} min/km")
    col2.metric("🏃 현재", f"{current:.1f} min/km",
                 delta=f"{'느림▲' if pace_diff>0 else '빠름▼'} "
                       f"{abs(pace_diff):.1f}",
                 delta_color="inverse")
    col3.metric("⏱️ 경과", f"{elapsed//60}분 {elapsed%60}초")

    st.markdown("---")
    st.subheader("🗺️ 실시간 위치 지도")
    if rd:
        path = rd["path"]
        cur_lat = st.session_state.current_lat or st.session_state.s_lat
        cur_lng = st.session_state.current_lng or st.session_state.s_lng
        m2 = folium.Map(location=[cur_lat, cur_lng], zoom_start=17)
        folium.PolyLine(path, color="#90CAF9",
                         weight=5, opacity=0.6).add_to(m2)
        track = st.session_state.gps_track
        if len(track) >= 2:
            folium.PolyLine([[t[0], t[1]] for t in track],
                             color="#FF6D00", weight=5,
                             opacity=0.95).add_to(m2)
        folium.CircleMarker([cur_lat, cur_lng], radius=10,
                             color="#1565C0", fill=True,
                             fill_color="#1E88E5",
                             fill_opacity=0.9,
                             tooltip="현재 위치").add_to(m2)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                       icon=folium.Icon(color="green",
                                         icon="play")).add_to(m2)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                       icon=folium.Icon(color="red",
                                         icon="flag")).add_to(m2)
        st_folium(m2, width=None, height=430)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔊 AI 상세 코칭", type="primary",
                      use_container_width=True):
            with st.spinner("AI 분석 중..."):
                t_lat = st.session_state.current_lat or st.session_state.s_lat
                t_lng = st.session_state.current_lng or st.session_state.s_lng
                w = get_weather_extended(t_lat, t_lng)
                risk, env_msgs, wbgt = get_environment_risk(
                    w["temp"], w["humidity"], w["wind_ms"])
                pace_status = "느리게" if pace_diff > 0 else "빠르게"
                cur_step = st.session_state.current_nav_step
                nav_desc = (steps[min(cur_step, len(steps)-1)]
                            ["description"] if steps else "직진")
                prompt = f"""
당신은 AI-Pacer 러닝 전문 코치입니다.
- 현재 내비: {nav_desc}
- 환경: {w['temp']}°C, 습도 {w['humidity']}%, WBGT {wbgt:.1f}°C,
        바람 {w['wind_ms']}m/s, {w['desc']}
- 환경 위험도: {risk}
- 페이스: 목표 {target}, 현재 {current} (목표보다 {abs(pace_diff):.1f} {pace_status})
- 경과: {elapsed//60}분 {elapsed%60}초

3문장 구어체로 코칭하세요:
1) 페이스 코칭
2) 신체효율/호흡 팁
3) 환경(WBGT/바람) 고려 조언
"""
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}])
                    msg = resp.choices[0].message.content
                    audio = client.audio.speech.create(
                        model="tts-1", voice="alloy", input=msg)
                    audio.stream_to_file("guide.mp3")
                    st.success(f"🤖 {msg}")
                    st.audio("guide.mp3")
                except Exception as e:
                    st.error(f"OpenAI 오류: {e}")
    with col_b:
        if st.button("🛑 러닝 종료", type="secondary",
                      use_container_width=True):
            st.session_state.running = False
            st.session_state.finished = True
            # ACWR 데이터 누적
            track = st.session_state.gps_track
            if track and len(track) >= 2:
                total_km = sum(
                    calc_distance_m(track[i][0], track[i][1],
                                     track[i+1][0], track[i+1][1])
                    for i in range(len(track)-1)) / 1000
                st.session_state.recent_runs.append(
                    (datetime.now().strftime("%Y-%m-%d"), total_km))
            st.rerun()


# ============================================================
# TAB 3: 결과 기록 + ACWR 업데이트 시각화
# ============================================================
with tab_result:
    if not st.session_state.finished and not st.session_state.gps_track:
        st.info("러닝을 완료하면 여기에 결과가 기록됩니다.")
        st.stop()

    st.subheader("🏅 러닝 완료 리포트")
    track = st.session_state.gps_track
    pace_hist = st.session_state.pace_history

    if track and len(track) >= 2:
        total_dist = sum(calc_distance_m(track[i][0], track[i][1],
                                          track[i+1][0], track[i+1][1])
                          for i in range(len(track)-1)) / 1000
        elapsed_total = track[-1][2] - track[0][2]
        avg_pace = (elapsed_total/60)/total_dist if total_dist > 0 else 0
        tgt = st.session_state.target_pace

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏃 총 거리",     f"{total_dist:.2f} km")
        col2.metric("⏱️ 총 시간",
                     f"{int(elapsed_total//60)}분 "
                     f"{int(elapsed_total%60)}초")
        col3.metric("📈 평균 페이스", f"{avg_pace:.1f} min/km")
        col4.metric("🎯 목표 대비",
                     f"{avg_pace-tgt:+.1f} min/km",
                     delta_color="inverse")

        # ACWR 변화 표시
        st.subheader("📊 이번 러닝의 ACWR 영향")
        if st.session_state.profile_set:
            today = datetime.now().date()
            daily = []
            if st.session_state.recent_runs:
                df_run = pd.DataFrame(
                    st.session_state.recent_runs,
                    columns=["date", "km"])
                df_run["date"] = pd.to_datetime(df_run["date"]).dt.date
                agg = df_run.groupby("date")["km"].sum().to_dict()
                for i in range(28, -1, -1):
                    d = today - timedelta(days=i)
                    daily.append(agg.get(d, 0.0))
            days_since = 0
            if st.session_state.recent_runs:
                first = pd.to_datetime(
                    st.session_state.recent_runs[0][0]).date()
                days_since = (today - first).days
            acwr, _, _, conf = calculate_acwr_ewma(
                daily, st.session_state.baseline_weekly_km, days_since)
            label, advice, status = get_acwr_advice(acwr, conf)
            if status == "success":
                st.success(f"{label} — {advice}")
            elif status == "warning":
                st.warning(f"{label} — {advice}")
            elif status == "error":
                st.error(f"{label} — {advice}")
            else:
                st.info(f"{label} — {advice}")

    st.subheader("🗺️ 러닝 발자취 지도")
    rd = st.session_state.route_data
    s_lat = st.session_state.s_lat
    if rd and s_lat:
        path = rd["path"]
        center_lat = ((track[0][0]+track[-1][0])/2
                       if track else s_lat)
        center_lng = ((track[0][1]+track[-1][1])/2
                       if track else st.session_state.s_lng)
        m3 = folium.Map(location=[center_lat, center_lng], zoom_start=15)
        folium.PolyLine(path, color="#9E9E9E",
                         weight=3, opacity=0.5).add_to(m3)
        if len(track) >= 2:
            for i in range(len(track)-1):
                ratio = i / (len(track)-1)
                r, g = int(255*ratio), int(255*(1-ratio))
                folium.PolyLine([[track[i][0],   track[i][1]],
                                  [track[i+1][0], track[i+1][1]]],
                                 color=f"#{r:02x}{g:02x}00",
                                 weight=5).add_to(m3)
        if track:
            folium.Marker([track[0][0],  track[0][1]],
                           icon=folium.Icon(color="green",
                                             icon="play")).add_to(m3)
            folium.Marker([track[-1][0], track[-1][1]],
                           icon=folium.Icon(color="red",
                                             icon="stop")).add_to(m3)
        st_folium(m3, width=None, height=500)

    if pace_hist:
        st.subheader("📈 페이스 변화 그래프")
        df = pd.DataFrame({
            "페이스 (min/km)": pace_hist,
            "목표": [st.session_state.target_pace]*len(pace_hist)
        })
        st.line_chart(df)

    if st.button("💾 기록 저장", use_container_width=True):
        record = {
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "route":        f"{st.session_state.s_name} → "
                            f"{st.session_state.e_name}",
            "track":        [(t[0], t[1]) for t in track],
            "pace_history": pace_hist,
            "target_pace":  st.session_state.target_pace,
            "user_level":   st.session_state.user_level,
            "baseline_km":  st.session_state.baseline_weekly_km,
        }
        st.download_button(
            "📥 JSON 다운로드",
            data=json.dumps(record, ensure_ascii=False, indent=2),
            file_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json")
