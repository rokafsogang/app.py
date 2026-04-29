"""
AI-Pacer Day 2 통합 버전
- WBGT 환경 분석, ACWR 부상추적, 페이스 윈도우, STT 부상확인
- LocalStorage 영구저장, st.secrets API키 분리
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

# LocalStorage (브라우저 영구 저장)
try:
    from streamlit_local_storage import LocalStorage
    HAS_LS = True
except ImportError:
    HAS_LS = False

# ===== API 키 (st.secrets) =====
OPENAI_API_KEY      = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
KAKAO_REST_API_KEY  = st.secrets["KAKAO_REST_API_KEY"]
TMAP_API_KEY        = st.secrets["TMAP_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="AI-Pacer", layout="wide")

# ============================================================
# Session State
# ============================================================
defaults = {
    "running": False, "finished": False,
    "gps_track": [], "pace_history": [],
    "start_time": None, "route_data": None,
    "s_lat": None, "s_lng": None,
    "e_lat": None, "e_lng": None,
    "s_name": "", "e_name": "",
    "current_lat": None, "current_lng": None,
    "current_pace": 0.0, "target_pace": 6.0,
    "current_nav_step": 0,
    "profile_set": False,
    "baseline_weekly_km": 0.0,
    "user_level": "입문",
    "recent_runs": [],
    "env_risk": None,
    "env_messages": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# LocalStorage 영구 저장
# ============================================================
DATA_KEY = "ai_pacer_data_v1"
if HAS_LS:
    localS = LocalStorage()
else:
    localS = None


def save_user_data():
    if not HAS_LS or localS is None:
        return False
    data = {
        "profile_set":        st.session_state.profile_set,
        "baseline_weekly_km": st.session_state.baseline_weekly_km,
        "user_level":         st.session_state.user_level,
        "recent_runs":        st.session_state.recent_runs,
    }
    try:
        localS.setItem(DATA_KEY,
                       json.dumps(data, ensure_ascii=False),
                       key=f"save_{datetime.now().timestamp()}")
        return True
    except Exception:
        return False


def load_user_data():
    if not HAS_LS or localS is None:
        return False
    try:
        raw = localS.getItem(DATA_KEY, key="load_userdata")
        if not raw:
            return False
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return False
        st.session_state.profile_set        = data.get("profile_set", False)
        st.session_state.baseline_weekly_km = data.get("baseline_weekly_km", 0.0)
        st.session_state.user_level         = data.get("user_level", "입문")
        st.session_state.recent_runs        = data.get("recent_runs", [])
        return True
    except Exception:
        return False


def clear_user_data():
    if HAS_LS and localS is not None:
        try:
            localS.deleteItem(DATA_KEY,
                              key=f"del_{datetime.now().timestamp()}")
        except Exception:
            pass
    for k in ["profile_set", "baseline_weekly_km",
              "user_level", "recent_runs"]:
        st.session_state[k] = defaults[k]


# 앱 시작 시 1회 로드
if "data_loaded" not in st.session_state:
    load_user_data()
    st.session_state.data_loaded = True

# ============================================================
# 1. 환경 분석 (WBGT)
# ============================================================
def calculate_wbgt(temp_c, humidity):
    e = (humidity / 100.0) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return 0.567 * temp_c + 0.393 * e + 3.94


def get_environment_risk(temp_c, humidity, wind_ms):
    wbgt = calculate_wbgt(temp_c, humidity)
    risk = "normal"
    msgs = []
    if wbgt >= 30:
        risk = "extreme"
        msgs.append(f"🚨 폭염 위험 (WBGT {wbgt:.1f}°C). 러닝 중단 권고.")
    elif wbgt >= 28:
        risk = "high"
        msgs.append(f"⚠️ 고온 경고 (WBGT {wbgt:.1f}°C). 페이스 10초/km 늦추고 5분마다 수분.")
    elif wbgt >= 23:
        risk = "moderate"
        msgs.append(f"☀️ 더위 주의 (WBGT {wbgt:.1f}°C). 페이스 5초/km 여유.")
    if temp_c <= -9:
        risk = "high"
        msgs.append(f"❄️ 한랭 경고 ({temp_c}°C). 워밍업 15분 이상 필수.")
    elif temp_c <= 0:
        if risk == "normal":
            risk = "moderate"
        msgs.append(f"❄️ 영하 ({temp_c}°C). 워밍업 10분, 첫 1km 페이스 30초 늦추세요.")
    elif temp_c <= 10:
        msgs.append(f"🧥 쌀쌀함 ({temp_c}°C). 워밍업 충분히.")
    if wind_ms >= 5.5:
        msgs.append(f"💨 강풍 ({wind_ms:.1f}m/s). 역풍 구간 페이스 10초/km 손해.")
    elif wind_ms >= 4.2:
        msgs.append(f"💨 바람 강함 ({wind_ms:.1f}m/s). 페이스 5초 여유.")
    if not msgs:
        msgs.append(f"✅ 쾌적 (WBGT {wbgt:.1f}°C, 바람 {wind_ms:.1f}m/s).")
    return risk, msgs, wbgt

# ============================================================
# 2. ACWR (EWMA)
# ============================================================
def calculate_acwr_ewma(daily_loads, baseline=0.0, days_since_signup=0):
    LA, LC = 0.25, 0.069
    baseline_daily = baseline / 7.0 if baseline > 0 else 0.0
    if not daily_loads and baseline_daily == 0:
        return None, 0, 0, "no_data"
    ewma_c = baseline_daily
    if daily_loads:
        recent_7 = daily_loads[-7:] if len(daily_loads) >= 7 else daily_loads
        ewma_a = sum(recent_7) / len(recent_7) if recent_7 else 0
    else:
        ewma_a = 0
    if days_since_signup < 7:
        conf = "self_report"
    elif days_since_signup < 28:
        conf = "mixed"
    else:
        conf = "measured"
    for load in daily_loads:
        ewma_a = load * LA + ewma_a * (1 - LA)
        ewma_c = load * LC + ewma_c * (1 - LC)
    if ewma_c < 0.01:
        return None, ewma_a, ewma_c, conf
    return ewma_a / ewma_c, ewma_a, ewma_c, conf


def estimate_baseline_load(freq_label, dist_label):
    freq_map = {"안 뜀": 0, "1-2회": 1.5, "3-4회": 3.5, "5회 이상": 5.5}
    dist_map = {"3km 이하": 2.5, "3-5km": 4.0, "5-10km": 7.0, "10km 이상": 12.0}
    return freq_map.get(freq_label, 0) * dist_map.get(dist_label, 0)


def get_acwr_advice(acwr, confidence):
    if acwr is None:
        return ("데이터 부족", "기초기록 설문을 완료해주세요.", "info")
    note = ""
    if confidence == "self_report":
        note = " (자가보고 추정)"
    elif confidence == "mixed":
        note = " (혼합)"
    if acwr < 0.8:
        return (f"⚠️ 저부하 ACWR {acwr:.2f}{note}",
                "천천히 시작하세요.", "warning")
    elif acwr <= 1.3:
        return (f"✅ Sweet Spot ACWR {acwr:.2f}{note}",
                "적정 훈련량입니다.", "success")
    elif acwr <= 1.5:
        return (f"⚠️ 주의 ACWR {acwr:.2f}{note}",
                "회복 챙기세요.", "warning")
    else:
        return (f"🚨 위험 ACWR {acwr:.2f}{note}",
                "부상 위험 높음. 회복 페이스 권장.", "error")

# ============================================================
# 외부 API
# ============================================================
def get_kakao_coords(address):
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    try:
        res = requests.get("https://dapi.kakao.com/v2/local/search/address.json",
                           headers=headers, params={"query": address}, timeout=5)
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            if docs:
                d = docs[0]
                return float(d['y']), float(d['x']), d.get('address_name', address)
        res2 = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json",
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


def get_tmap_walking_route(s_lat, s_lng, e_lat, e_lng, s_name="출발지", e_name="목적지"):
    url = f"https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&format=json&appKey={TMAP_API_KEY}"
    body = {
        "startX": str(s_lng), "startY": str(s_lat),
        "endX": str(e_lng), "endY": str(e_lat),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO",
        "startName": s_name, "endName": e_name, "searchOption": "0",
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=15)
        if r.status_code != 200:
            return None, f"Tmap 오류 {r.status_code}"
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
                    steps.append({"description": desc, "distance": float(dist),
                                  "pointType": pt, "lat": coord[1], "lng": coord[0]})
        summary = features[0]["properties"]
        return {"path": path, "steps": steps,
                "dist_m": summary.get("totalDistance", 0),
                "time_sec": summary.get("totalTime", 0)}, None
    except Exception as e:
        return None, str(e)


def get_weather_extended(lat, lng):
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lng}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr")
        res = requests.get(url, timeout=5).json()
        return {
            "temp": res['main']['temp'], "humidity": res['main']['humidity'],
            "desc": res['weather'][0]['description'],
            "wind_ms": res.get('wind', {}).get('speed', 0),
            "wind_deg": res.get('wind', {}).get('deg', 0),
        }
    except Exception:
        return {"temp": 20.0, "humidity": 50, "desc": "알 수 없음", "wind_ms": 0, "wind_deg": 0}


def calc_distance_m(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ============================================================
# 3. JS 컴포넌트 (GPS + 페이스윈도우 + STT + 구간멘트)
# ============================================================
def build_auto_component(steps, target_pace, total_dist_m, env_messages):
    sj = json.dumps(steps, ensure_ascii=False)
    ej = json.dumps(env_messages, ensure_ascii=False)
    return f"""
<div style="background:#1a1a2e;border-radius:10px;padding:12px;color:#e0e0e0;font-family:monospace;font-size:13px;border:1px solid #333;">
  <div id="gps-info">📡 GPS 초기화 중...</div>
  <div id="nav-info" style="margin-top:8px;font-size:15px;font-weight:bold;color:#64b5f6;">🧭 대기 중</div>
  <div id="pace-info" style="margin-top:6px;color:#a5d6a7;">⚡ 페이스 측정 중...</div>
  <div id="zone-info" style="margin-top:6px;color:#ffb74d;">📍 진행률 계산 중...</div>
  <div id="pain-check" style="margin-top:8px;padding:8px;background:#3e2723;border-radius:6px;display:none;color:#ffccbc;font-weight:bold;"></div>
  <div id="speech-log" style="margin-top:6px;font-size:11px;color:#888;"></div>
</div>
<script>
const STEPS={sj},TARGET_PACE={target_pace},TOTAL_DIST={total_dist_m},ENV_MESSAGES={ej};
const WAYPOINT_RADIUS=30,SUDDEN_DROP=0.08,GRADUAL_DROP=0.05;
const COOLDOWN={{emergency:0,injury:5000,pace:30000,motivation:60000,env:120000}};
let currentStep=0,lastNavSpeak=0,lastByCat={{pace:0,motivation:0,injury:0,env:0}};
let painCheckActive=false,painCheckStart=0;
let paceLog=[],totalDist=0,lastLat=null,lastLng=null;let gpsTrack=[];
const KW={{safe:['괜찮','문제없','오케이','ok','좋아'],injury:['무릎','발목','종아리','햄스트링','허벅지','발바닥','허리','아파','시려','땡겨','저려','쥐'],emergency:['가슴','숨','어지','토할','쓰러'],fatigue:['힘들','지쳐','못하겠']}};

function speak(t,cat){{if(!window.speechSynthesis)return;const now=Date.now();if(cat&&lastByCat[cat]!==undefined){{if(now-lastByCat[cat]<COOLDOWN[cat])return;lastByCat[cat]=now;}}if(cat==='emergency'||cat==='injury')window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.lang='ko-KR';u.rate=1.0;window.speechSynthesis.speak(u);document.getElementById('speech-log').innerText='🔊 '+t;}}
function hav(a1,n1,a2,n2){{const R=6371000,dL=(a2-a1)*Math.PI/180,dN=(n2-n1)*Math.PI/180,a=Math.sin(dL/2)**2+Math.cos(a1*Math.PI/180)*Math.cos(a2*Math.PI/180)*Math.sin(dN/2)**2;return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));}}
function wavg(sec){{const c=Date.now()-sec*1000,r=paceLog.filter(p=>p.ts>=c&&p.pace>0);if(!r.length)return 0;return r.reduce((s,p)=>s+p.pace,0)/r.length;}}
function detectPattern(){{const w1=wavg(60),w5=wavg(300);if(!w1||!w5)return'init';const s=(w1-w5)/w5;if(s>SUDDEN_DROP)return'sudden_drop';if(s<-SUDDEN_DROP)return'sudden_spike';if(paceLog.length>20){{const bl=paceLog.slice(0,10).filter(p=>p.pace>0);const bv=bl.reduce((s,p)=>s+p.pace,0)/Math.max(1,bl.length);if(bv>0&&(w5-bv)/bv>GRADUAL_DROP)return'gradual_drop';}}return'stable';}}

let recog=null;
function startPainCheck(){{if(painCheckActive)return;painCheckActive=true;painCheckStart=Date.now();const box=document.getElementById('pain-check');box.style.display='block';box.innerText='🎙️ 응답 대기 중...';speak('페이스가 떨어졌어요. 어디 불편한 곳 있나요? 괜찮으면 괜찮아라고 답해주세요.','injury');if(!('webkitSpeechRecognition' in window)&&!('SpeechRecognition' in window)){{box.innerText='⚠️ 음성인식 미지원';return;}}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;recog=new SR();recog.lang='ko-KR';recog.interimResults=false;recog.onresult=(e)=>{{const t=e.results[0][0].transcript.toLowerCase();box.innerText='👂 '+t;handlePain(t);}};recog.onerror=()=>{{box.innerText='⚠️ 음성 인식 오류';painCheckActive=false;}};recog.onend=()=>{{if(painCheckActive&&Date.now()-painCheckStart>15000){{box.innerText='⏱️ 응답없음';speak('응답이 없어요. 페이스 줄이고 걸어주세요.','injury');painCheckActive=false;}}}};try{{recog.start();}}catch(e){{}}}}

function handlePain(t){{const box=document.getElementById('pain-check');for(const k of KW.emergency){{if(t.includes(k)){{box.style.background='#b71c1c';box.innerText='🚨 응급: '+k;speak('즉시 멈추고 안전한 곳에 앉으세요. 119 연락하세요.','emergency');painCheckActive=false;return;}}}}for(const k of KW.injury){{if(t.includes(k)){{box.style.background='#bf360c';box.innerText='⚠️ 부상: '+k;speak('지금 바로 멈추세요. 통증 무시하면 큰 부상으로 이어져요.','injury');painCheckActive=false;return;}}}}for(const k of KW.safe){{if(t.includes(k)){{box.style.background='#1b5e20';box.innerText='✅ 안전 확인';speak('다행이에요. 호흡 가다듬고 회복해봐요.','pace');painCheckActive=false;return;}}}}for(const k of KW.fatigue){{if(t.includes(k)){{box.innerText='😮‍💨 피로';speak('페이스 늦추고 4초 호흡으로 바꿔보세요.','pace');painCheckActive=false;return;}}}}box.innerText='❓ 미감지: '+t;painCheckActive=false;}}

function zoneMsg(prog,ps){{if(prog<0.3)return ps==='fast'?'초반 너무 빨라요. 후반 위해 늦추세요.':'좋은 시작이에요. 호흡 안정시키며 가요.';if(prog<0.7)return ps==='slow'?'중반이에요. 자세 점검, 어깨 힘 빼세요.':'리듬 좋아요. 유지하세요.';if(prog<0.9)return'끝이 보여요. 무너지면 아깝잖아요.';return'마지막! 끝까지 갑시다!';}}

function onGPS(pos){{const lat=pos.coords.latitude,lng=pos.coords.longitude,spd=pos.coords.speed,now=Date.now();if(lastLat!==null){{const seg=hav(lastLat,lastLng,lat,lng);if(seg<50)totalDist+=seg;}}lastLat=lat;lastLng=lng;gpsTrack.push([lat,lng,now/1000]);if(gpsTrack.length%3===0){{try{{localStorage.setItem('ai_pacer_live_run',JSON.stringify({{track:gpsTrack,paces:paceLog.map(p=>p.pace),totalDist:totalDist}}));}}catch(e){{}}}};let pace=0;if(spd&&spd>0.3)pace=(1000/spd)/60;if(pace>0)paceLog.push({{ts:now,pace}});if(paceLog.length>600)paceLog.shift();const ps=pace>0?Math.floor(pace)+'분 '+Math.round((pace%1)*60)+'초/km':'측정중';document.getElementById('gps-info').innerText='✅ GPS | '+lat.toFixed(5)+', '+lng.toFixed(5)+' | '+ps+' | '+Math.round(totalDist)+'m';const prog=TOTAL_DIST>0?Math.min(1,totalDist/TOTAL_DIST):0;document.getElementById('zone-info').innerText='📍 '+( prog*100).toFixed(1)+'% | '+Math.round(totalDist)+'m / '+Math.round(TOTAL_DIST)+'m';

if(currentStep<STEPS.length){{const step=STEPS[currentStep];const d=hav(lat,lng,step.lat,step.lng);document.getElementById('nav-info').innerText='🧭 ['+(currentStep+1)+'/'+STEPS.length+'] '+(step.distance>0?Math.round(step.distance)+'m ':'') +step.description;if(d<WAYPOINT_RADIUS&&now-lastNavSpeak>5000){{lastNavSpeak=now;speak(step.description,null);if(currentStep+1<STEPS.length)currentStep++;else speak('목적지 도착! 수고하셨습니다!',null);}}}}

if(!painCheckActive&&pace>0){{const pat=detectPattern();let pc='stable';if(pat==='sudden_drop'){{startPainCheck();pc='slow';}}else if(pat==='gradual_drop'){{const dk=totalDist/1000;let m='페이스가 떨어지고 있어요. ';if(dk<5)m+='호흡 가다듬으세요.';else if(dk<15)m+='자세 점검. 어깨 힘 빼세요.';else m+='에너지 보충하세요. 젤이나 당분.';speak(m,'pace');pc='slow';}}else if(pat==='sudden_spike'){{speak('너무 빨라졌어요. 원래 페이스로.','pace');pc='fast';}}else{{const diff=(pace-TARGET_PACE)/TARGET_PACE;if(diff>0.10){{speak('목표보다 느려요. 올려보세요.','pace');pc='slow';}}else if(diff<-0.10){{speak('너무 빨라요. 줄이세요.','pace');pc='fast';}}}}const info=document.getElementById('pace-info');if(pc==='slow'){{info.style.color='#ef9a9a';info.innerText='🐢 느림 — '+ps;}}else if(pc==='fast'){{info.style.color='#fff176';info.innerText='🚀 빠름 — '+ps;}}else{{info.style.color='#a5d6a7';info.innerText='✅ 정상 — '+ps;}}if(now-lastByCat.motivation>COOLDOWN.motivation){{speak(zoneMsg(prog,pc),'motivation');}}}}

try{{window.parent.postMessage({{type:'streamlit:setComponentValue',value:JSON.stringify({{lat:lastLat,lng:lastLng,pace:0,step:currentStep,dist:totalDist,ts:Date.now()}})}},'*');}}catch(e){{}}}}

if(navigator.geolocation){{setTimeout(()=>speak('AI 페이서 시작합니다.',null),500);ENV_MESSAGES.forEach((m,i)=>{{setTimeout(()=>speak(m,'env'),2000+i*4500);}});if(STEPS.length>0)setTimeout(()=>speak('첫 안내. '+STEPS[0].description,null),2000+ENV_MESSAGES.length*4500);navigator.geolocation.watchPosition(onGPS,err=>{{document.getElementById('gps-info').innerText='❌ GPS 오류: '+err.message;}},{{enableHighAccuracy:true,timeout:5000,maximumAge:0}});}}else{{document.getElementById('gps-info').innerText='❌ GPS 미지원';}}
</script>
"""

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.header("👤 사용자 프로필")
    if HAS_LS:
        st.caption("💾 데이터는 이 브라우저에 영구 저장됩니다")
    else:
        st.caption("⚠️ 새로고침 시 데이터 사라짐 (streamlit-local-storage 미설치)")

    if not st.session_state.profile_set:
        st.warning("초기 프로필을 설정해주세요")

    with st.expander("🎯 기초기록 설문", expanded=not st.session_state.profile_set):
        st.caption("ACSM Pre-participation 가이드 기반")
        freq = st.selectbox("최근 한 달 주간 러닝 빈도",
                             ["안 뜀", "1-2회", "3-4회", "5회 이상"])
        dist = st.selectbox("1회당 평균 거리",
                             ["3km 이하", "3-5km", "5-10km", "10km 이상"])
        level = st.radio("러닝 레벨",
                          ["입문 (3개월 미만)", "초급 (6개월)",
                           "중급 (1년+)", "상급 (마라톤 경험)"])
        if st.button("프로필 저장", use_container_width=True):
            baseline = estimate_baseline_load(freq, dist)
            st.session_state.baseline_weekly_km = baseline
            st.session_state.user_level = level
            st.session_state.profile_set = True
            save_user_data()
            st.success(f"✅ 주간 {baseline:.1f}km (저장됨)")
            st.rerun()

    st.markdown("---")
    st.subheader("📊 ACWR 부상위험")

    today = datetime.now().date()
    daily = []
    if st.session_state.recent_runs:
        df_run = pd.DataFrame(st.session_state.recent_runs, columns=["date", "km"])
        df_run["date"] = pd.to_datetime(df_run["date"]).dt.date
        agg = df_run.groupby("date")["km"].sum().to_dict()
        for i in range(28, -1, -1):
            d = today - timedelta(days=i)
            daily.append(agg.get(d, 0.0))

    days_since = 0
    if st.session_state.recent_runs:
        first = pd.to_datetime(st.session_state.recent_runs[0][0]).date()
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
        st.metric("Acute (7일)", f"{ewma_a*7:.1f} km")
        st.metric("Chronic (28일)", f"{ewma_c*7:.1f} km")
        st.caption("Sweet Spot: 0.8 ≤ ACWR ≤ 1.3 (Gabbett 2016)")

    st.markdown("---")
    with st.expander("📚 누적 기록", expanded=False):
        if st.session_state.recent_runs:
            df_r = pd.DataFrame(st.session_state.recent_runs, columns=["날짜", "km"])
            df_r["km"] = df_r["km"].round(1)
            st.dataframe(df_r, hide_index=True, use_container_width=True)
            st.caption(f"총 {len(st.session_state.recent_runs)}회")
        else:
            st.caption("기록 없음")
        if st.button("🗑️ 데이터 초기화", use_container_width=True):
            clear_user_data()
            st.rerun()

    st.markdown("---")
    with st.expander("🎬 발표 시연용 데모", expanded=False):
        if st.button("✅ Sweet Spot 패턴", use_container_width=True):
            runs = []
            for i in range(28, 0, -1):
                if i % 2 == 0:
                    runs.append(((today - timedelta(days=i)).strftime("%Y-%m-%d"), 5.0))
            st.session_state.recent_runs = runs
            save_user_data()
            st.rerun()
        if st.button("🚨 Danger Zone 패턴", use_container_width=True):
            runs = []
            for i in range(28, 7, -1):
                if i % 4 == 0:
                    runs.append(((today - timedelta(days=i)).strftime("%Y-%m-%d"), 3.0))
            for i in range(7, 0, -1):
                runs.append(((today - timedelta(days=i)).strftime("%Y-%m-%d"), 12.0))
            st.session_state.recent_runs = runs
            save_user_data()
            st.rerun()

# ============================================================
# 메인 UI
# ============================================================
st.title("🏃‍♂️ AI-Pacer")
tab_setup, tab_running, tab_result = st.tabs(["⚙️ 경로 설정", "🏃 러닝 중", "📊 결과"])

# TAB 1
with tab_setup:
    st.subheader("📍 경로 및 목표 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_input = st.text_input("출발지", placeholder="예: 서강대학교")
        target_pace = st.slider("목표 페이스 (min/km)", 2.0, 12.0,
                                 st.session_state.target_pace, step=0.1)
        st.session_state.target_pace = target_pace
        pm, ps = int(target_pace), int((target_pace % 1) * 60)
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
                with st.spinner("경로 계산 중..."):
                    route, err = get_tmap_walking_route(s_lat, s_lng, e_lat, e_lng, s_name, e_name)
                if err:
                    st.error(f"경로 오류: {err}")
                else:
                    st.session_state.update({
                        "s_lat": s_lat, "s_lng": s_lng, "s_name": s_name,
                        "e_lat": e_lat, "e_lng": e_lng, "e_name": e_name,
                        "route_data": route})
                    st.success(f"✅ {s_name} → {e_name}")

    if st.session_state.route_data:
        rd = st.session_state.route_data
        dist_km = rd["dist_m"] / 1000
        est_min = dist_km * st.session_state.target_pace
        weather = get_weather_extended(st.session_state.s_lat, st.session_state.s_lng)
        risk, env_msgs, wbgt = get_environment_risk(
            weather["temp"], weather["humidity"], weather["wind_ms"])
        st.session_state.env_risk = risk
        st.session_state.env_messages = env_msgs

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 거리", f"{dist_km:.2f} km")
        c2.metric("예상 완주", f"{int(est_min)}분 {int((est_min%1)*60)}초")
        c3.metric("기온/습도", f"{weather['temp']:.0f}°C / {weather['humidity']}%",
                  delta=weather['desc'], delta_color="off")
        c4.metric("WBGT", f"{wbgt:.1f}°C",
                  delta=f"바람 {weather['wind_ms']:.1f}m/s", delta_color="off")

        st.subheader("🌤️ 환경 분석")
        for msg in env_msgs:
            if risk == "extreme":
                st.error(msg)
            elif risk == "high":
                st.warning(msg)
            elif risk == "moderate":
                st.info(msg)
            else:
                st.success(msg)
        st.caption("근거: ACSM Guidelines, Beaumont & Polidori (2025)")

        path = rd["path"]
        mp = folium.Map(location=[(st.session_state.s_lat+st.session_state.e_lat)/2,
                                   (st.session_state.s_lng+st.session_state.e_lng)/2], zoom_start=15)
        folium.PolyLine(path, color="#1E88E5", weight=6, opacity=0.9).add_to(mp)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                      tooltip=f"출발: {st.session_state.s_name}",
                      icon=folium.Icon(color="green", icon="play")).add_to(mp)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                      tooltip=f"도착: {st.session_state.e_name}",
                      icon=folium.Icon(color="red", icon="flag")).add_to(mp)
        st_folium(mp, width=None, height=420)

        with st.expander("🧭 전체 경로 안내"):
            for i, s in enumerate(rd["steps"]):
                ds = f"  ({int(s['distance'])}m)" if s['distance'] else ""
                st.write(f"**{i+1}.** {s['description']}{ds}")

        if not st.session_state.profile_set:
            st.warning("⚠️ 사이드바에서 기초기록 설문 먼저 완료해주세요.")

        if st.button("🏃 러닝 시작!", type="primary", use_container_width=True):
            st.session_state.update({
                "running": True, "finished": False,
                "gps_track": [], "pace_history": [],
                "start_time": datetime.now(), "current_nav_step": 0})
            st.success("러닝 시작! '러닝 중' 탭으로 이동하세요.")

# TAB 2
with tab_running:
    if not st.session_state.running and not st.session_state.finished:
        st.info("경로 설정 탭에서 러닝을 시작하세요.")
        st.stop()
    if st.session_state.finished:
        st.success("러닝 종료. 결과 탭을 확인하세요.")
        st.stop()

    rd = st.session_state.route_data
    steps = rd["steps"] if rd else []
    total_m = rd["dist_m"] if rd else 0

    st.subheader("📡 실시간 안내 + 부상 모니터링")
    st.caption("🔊 폰 음소거 해제 + 마이크 권한 필수")
    components.html(build_auto_component(
        steps, st.session_state.target_pace,
        total_m, st.session_state.env_messages or []), height=210)

    st.markdown("---")
    target = st.session_state.target_pace
    current = st.session_state.current_pace
    if current and current > 0:
        pace_diff = current - target
    else:
        current = 0
        pace_diff = 0
    elapsed = int((datetime.now() - st.session_state.start_time).total_seconds()) \
        if st.session_state.start_time else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 목표", f"{target:.1f} min/km")
    c2.metric("🏃 현재", f"{current:.1f} min/km",
              delta=f"{'느림▲' if pace_diff>0 else '빠름▼'} {abs(pace_diff):.1f}",
              delta_color="inverse")
    c3.metric("⏱️ 경과", f"{elapsed//60}분 {elapsed%60}초")

    st.markdown("---")
    if rd:
        path = rd["path"]
        cl = st.session_state.current_lat or st.session_state.s_lat
        cn = st.session_state.current_lng or st.session_state.s_lng
        m2 = folium.Map(location=[cl, cn], zoom_start=17)
        folium.PolyLine(path, color="#90CAF9", weight=5, opacity=0.6).add_to(m2)
        track = st.session_state.gps_track
        if len(track) >= 2:
            folium.PolyLine([[t[0], t[1]] for t in track],
                             color="#FF6D00", weight=5, opacity=0.95).add_to(m2)
        folium.CircleMarker([cl, cn], radius=10, color="#1565C0",
                             fill=True, fill_color="#1E88E5", fill_opacity=0.9).add_to(m2)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                       icon=folium.Icon(color="green", icon="play")).add_to(m2)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                       icon=folium.Icon(color="red", icon="flag")).add_to(m2)
        st_folium(m2, width=None, height=430)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        if st.button("🔊 AI 상세 코칭", type="primary", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                tl = st.session_state.current_lat or st.session_state.s_lat
                tn = st.session_state.current_lng or st.session_state.s_lng
                w = get_weather_extended(tl, tn)
                r2, e2, wb = get_environment_risk(w["temp"], w["humidity"], w["wind_ms"])
                ps = "느리게" if pace_diff > 0 else "빠르게"
                cs = st.session_state.current_nav_step
                nd = steps[min(cs, len(steps)-1)]["description"] if steps else "직진"
                prompt = f"""AI-Pacer 러닝 코치.
내비: {nd}
환경: {w['temp']}°C, 습도 {w['humidity']}%, WBGT {wb:.1f}°C, 바람 {w['wind_ms']}m/s, {w['desc']}
위험도: {r2}
페이스: 목표 {target}, 현재 {current} (목표보다 {abs(pace_diff):.1f} {ps})
경과: {elapsed//60}분 {elapsed%60}초
3문장 구어체: 페이스코칭 → 호흡팁 → 환경조언"""
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
                    msg = resp.choices[0].message.content
                    audio = client.audio.speech.create(model="tts-1", voice="alloy", input=msg)
                    audio.stream_to_file("guide.mp3")
                    st.success(f"🤖 {msg}")
                    st.audio("guide.mp3")
                except Exception as e:
                    st.error(f"OpenAI 오류: {e}")
    with cb:
        if st.button("🛑 러닝 종료", type="secondary", use_container_width=True):
            st.session_state.running = False
            st.session_state.finished = True
            st.rerun()

# TAB 3
with tab_result:
    if not st.session_state.finished:
        st.info("러닝을 완료하면 여기에 결과가 기록됩니다.")
        st.stop()

    # JS가 localStorage에 저장한 GPS 데이터 읽기
    run_data = None
    if HAS_LS and localS is not None:
        try:
            raw = localS.getItem("ai_pacer_live_run", key="get_run_result")
            if raw:
                run_data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

    if run_data and isinstance(run_data, dict):
        track = run_data.get("track", [])
        pace_list = run_data.get("paces", [])
        total_dist_m = run_data.get("totalDist", 0)
    else:
        track = []
        pace_list = []
        total_dist_m = 0

    st.subheader("🏅 러닝 리포트")
    track = st.session_state.gps_track
    pace_hist = st.session_state.pace_history

    st.subheader("🏅 러닝 리포트")
    if track and len(track) >= 2:
        total_dist = total_dist_m / 1000
        elapsed_total = track[-1][2] - track[0][2]
        avg_pace = (elapsed_total / 60) / total_dist if total_dist > 0 else 0
        tgt = st.session_state.target_pace

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏃 총 거리", f"{total_dist:.1f} km")
        c2.metric("⏱️ 시간", f"{int(elapsed_total//60)}분 {int(elapsed_total%60)}초")
        c3.metric("📈 평균 페이스", f"{avg_pace:.1f} min/km")
        c4.metric("🎯 목표 대비", f"{avg_pace-tgt:+.1f} min/km", delta_color="inverse")

        # ACWR에 기록 추가
        if total_dist > 0.1:
            already_saved = False
            today_str = datetime.now().strftime("%Y-%m-%d")
            for r in st.session_state.recent_runs:
                if r[0] == today_str and abs(r[1] - round(total_dist, 1)) < 0.01:
                    already_saved = True
            if not already_saved:
                st.session_state.recent_runs.append((today_str, round(total_dist, 1)))
                save_user_data()
    else:
        st.warning("GPS 데이터를 불러오는 중입니다. 페이지를 한 번 새로고침 해주세요.")
        avg_pace = (elapsed_total/60)/total_dist if total_dist > 0 else 0
        tgt = st.session_state.target_pace

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏃 총 거리", f"{total_dist:.1f} km")
        c2.metric("⏱️ 시간", f"{int(elapsed_total//60)}분 {int(elapsed_total%60)}초")
        c3.metric("📈 평균 페이스", f"{avg_pace:.1f} min/km")
        c4.metric("🎯 목표 대비", f"{avg_pace-tgt:+.1f} min/km", delta_color="inverse")

        st.subheader("📊 ACWR 영향")
        if st.session_state.profile_set:
            td = datetime.now().date()
            dl = []
            if st.session_state.recent_runs:
                dfr = pd.DataFrame(st.session_state.recent_runs, columns=["date", "km"])
                dfr["date"] = pd.to_datetime(dfr["date"]).dt.date
                ag = dfr.groupby("date")["km"].sum().to_dict()
                for i in range(28, -1, -1):
                    dl.append(ag.get(td - timedelta(days=i), 0.0))
            ds = 0
            if st.session_state.recent_runs:
                ds = (td - pd.to_datetime(st.session_state.recent_runs[0][0]).date()).days
            a2, _, _, c2 = calculate_acwr_ewma(dl, st.session_state.baseline_weekly_km, ds)
            l2, ad2, s2 = get_acwr_advice(a2, c2)
            if s2 == "success":
                st.success(f"{l2} — {ad2}")
            elif s2 == "warning":
                st.warning(f"{l2} — {ad2}")
            elif s2 == "error":
                st.error(f"{l2} — {ad2}")
            else:
                st.info(f"{l2} — {ad2}")

    st.subheader("🗺️ 러닝 발자취")
    rd = st.session_state.route_data
    if rd and st.session_state.s_lat:
        path = rd["path"]
        clat = (track[0][0]+track[-1][0])/2 if track else st.session_state.s_lat
        clng = (track[0][1]+track[-1][1])/2 if track else st.session_state.s_lng
        m3 = folium.Map(location=[clat, clng], zoom_start=15)
        folium.PolyLine(path, color="#9E9E9E", weight=3, opacity=0.5).add_to(m3)
        if len(track) >= 2:
            for i in range(len(track)-1):
                ratio = i / (len(track)-1)
                r, g = int(255*ratio), int(255*(1-ratio))
                folium.PolyLine([[track[i][0], track[i][1]],
                                  [track[i+1][0], track[i+1][1]]],
                                 color=f"#{r:02x}{g:02x}00", weight=5).add_to(m3)
        if track:
            folium.Marker([track[0][0], track[0][1]],
                           icon=folium.Icon(color="green", icon="play")).add_to(m3)
            folium.Marker([track[-1][0], track[-1][1]],
                           icon=folium.Icon(color="red", icon="stop")).add_to(m3)
        st_folium(m3, width=None, height=500)

    if pace_list:
        st.subheader("📈 페이스 변화")
        df = pd.DataFrame({"페이스": pace_list,
                            "목표": [st.session_state.target_pace] * len(pace_list)})
        st.line_chart(df)

    if st.button("💾 기록 저장", use_container_width=True):
        record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "route": f"{st.session_state.s_name} → {st.session_state.e_name}",
            "track": [(t[0], t[1]) for t in track],
            "pace_history": pace_hist,
            "target_pace": st.session_state.target_pace,
        }
        st.download_button("📥 JSON 다운로드",
                            data=json.dumps(record, ensure_ascii=False, indent=2),
                            file_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json")
