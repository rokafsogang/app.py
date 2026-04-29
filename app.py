import streamlit as st
import streamlit.components.v1 as components
import requests
import folium
import json
import time
import math
import pandas as pd
from streamlit_folium import st_folium
from openai import OpenAI
from datetime import datetime

# ===== API 키 (Streamlit Secrets에서 불러오기) =====
OPENAI_API_KEY      = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
KAKAO_REST_API_KEY  = st.secrets["KAKAO_REST_API_KEY"]
TMAP_API_KEY        = st.secrets["TMAP_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
st.set_page_config(page_title="AI-Pacer", layout="wide")

# ===== Session State =====
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# API 함수
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
        "endX":   str(e_lng), "endY":   str(e_lat),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO",
        "startName": s_name, "endName": e_name,
        "searchOption": "0",
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          json=body, timeout=15)
        if r.status_code != 200:
            return None, f"Tmap API 오류 {r.status_code}: {r.text[:300]}"
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
                desc  = props.get("description", "")
                dist  = props.get("distance", 0)
                pt    = props.get("pointType", "")
                coord = f["geometry"]["coordinates"]
                if pt not in ("SP",) and desc:
                    steps.append({
                        "description": desc,
                        "distance":    float(dist),
                        "pointType":   pt,
                        "lat":         coord[1],
                        "lng":         coord[0],
                    })

        summary = features[0]["properties"]
        return {
            "path":     path,
            "steps":    steps,
            "dist_m":   summary.get("totalDistance", 0),
            "time_sec": summary.get("totalTime", 0),
        }, None
    except Exception as e:
        return None, str(e)


def get_weather(lat, lng):
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lng}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr")
        res = requests.get(url, timeout=5).json()
        return res['main']['temp'], res['main']['humidity'], res['weather'][0]['description']
    except:
        return 20.0, 50, "알 수 없음"


def calc_distance_m(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lng2-lng1)/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ============================================================
# 자동 GPS + 자동 턴바이턴 + 자동 페이스 TTS 컴포넌트
# ============================================================
def build_auto_component(steps, target_pace):
    steps_json = json.dumps(steps, ensure_ascii=False)
    return f"""
<div style="background:#1a1a2e; border-radius:10px; padding:12px; color:#e0e0e0;
            font-family:monospace; font-size:13px; border:1px solid #333;">
  <div id="gps-info">📡 GPS 초기화 중...</div>
  <div id="nav-info" style="margin-top:8px; font-size:15px; font-weight:bold; color:#64b5f6;">
    🧭 내비게이션 대기 중...
  </div>
  <div id="pace-info" style="margin-top:6px; color:#a5d6a7;">⚡ 페이스 측정 중...</div>
  <div id="speech-log" style="margin-top:6px; font-size:11px; color:#888;"></div>
</div>

<script>
const STEPS           = {steps_json};
const TARGET_PACE     = {target_pace};
const ALERT_THRESHOLD = 0.10;
const WAYPOINT_RADIUS = 30;

let currentStep   = 0;
let lastPaceAlert = 0;
let lastNavSpeak  = 0;
const PACE_COOLDOWN = 30000;
const NAV_COOLDOWN  = 5000;

function speak(text) {{
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
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
          + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLng/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

function sendToStreamlit(lat, lng, pace, step) {{
  try {{
    window.parent.postMessage({{
      type: 'streamlit:setComponentValue',
      value: JSON.stringify({{lat, lng, pace, step, ts: Date.now()}})
    }}, '*');
  }} catch(e) {{}}
}}

function onGPS(pos) {{
  const lat = pos.coords.latitude;
  const lng = pos.coords.longitude;
  const spd = pos.coords.speed;
  const now = Date.now();

  let pace = 0;
  if (spd && spd > 0.3) pace = (1000 / spd) / 60;

  const paceStr = pace > 0
    ? Math.floor(pace) + '분 ' + Math.round((pace%1)*60) + '초/km'
    : '측정중';

  document.getElementById('gps-info').innerText =
    '✅ GPS | 위도 ' + lat.toFixed(5) + ' | 경도 ' + lng.toFixed(5) + ' | 페이스 ' + paceStr;

  // ── 자동 턴바이턴 ──
  if (currentStep < STEPS.length) {{
    const step = STEPS[currentStep];
    const dist = haversine(lat, lng, step.lat, step.lng);

    document.getElementById('nav-info').innerText =
      '🧭 [' + (currentStep+1) + '/' + STEPS.length + '] '
      + (step.distance > 0 ? Math.round(step.distance) + 'm 앞  ' : '')
      + step.description;

    if (dist < WAYPOINT_RADIUS && now - lastNavSpeak > NAV_COOLDOWN) {{
      lastNavSpeak = now;
      speak(step.description);
      if (currentStep + 1 < STEPS.length) {{
        currentStep++;
        const next = STEPS[currentStep];
        setTimeout(() => {{
          document.getElementById('nav-info').innerText =
            '🧭 [' + (currentStep+1) + '/' + STEPS.length + '] '
            + (next.distance > 0 ? Math.round(next.distance) + 'm 앞  ' : '')
            + next.description;
        }}, 2000);
      }} else {{
        speak('목적지에 도착했습니다. 수고하셨습니다!');
        document.getElementById('nav-info').innerText = '🏁 목적지 도착!';
      }}
    }}
  }}

  // ── 자동 페이스 TTS (±10%) ──
  if (pace > 0 && now - lastPaceAlert > PACE_COOLDOWN) {{
    const diff    = (pace - TARGET_PACE) / TARGET_PACE;
    const pMin    = Math.floor(pace);
    const pSec    = Math.round((pace%1)*60);
    const tMin    = Math.floor(TARGET_PACE);
    const tSec    = Math.round((TARGET_PACE%1)*60);

    if (diff > 0.10) {{
      lastPaceAlert = now;
      speak('페이스가 너무 느립니다. 현재 ' + pMin + '분 ' + pSec + '초, 목표는 ' + tMin + '분 ' + tSec + '초입니다. 속도를 높이세요.');
      document.getElementById('pace-info').style.color = '#ef9a9a';
      document.getElementById('pace-info').innerText = '🐢 느림 — ' + paceStr;
    }} else if (diff < -0.10) {{
      lastPaceAlert = now;
      speak('페이스가 너무 빠릅니다. 현재 ' + pMin + '분 ' + pSec + '초. 후반을 위해 속도를 줄이세요.');
      document.getElementById('pace-info').style.color = '#fff176';
      document.getElementById('pace-info').innerText = '🚀 빠름 — ' + paceStr;
    }} else {{
      document.getElementById('pace-info').style.color = '#a5d6a7';
      document.getElementById('pace-info').innerText = '✅ 페이스 정상 — ' + paceStr;
    }}
  }}

  sendToStreamlit(lat, lng, pace, currentStep);
}}

if (navigator.geolocation) {{
  setTimeout(() => speak('AI 페이서 시작합니다. 경로 안내를 시작합니다.'), 1000);
  if (STEPS.length > 0) {{
    setTimeout(() => speak(STEPS[0].description), 3000);
  }}
  navigator.geolocation.watchPosition(onGPS,
    err => {{
      document.getElementById('gps-info').innerText = '❌ GPS 오류: ' + err.message;
    }},
    {{enableHighAccuracy: true, timeout: 5000, maximumAge: 0}}
  );
}} else {{
  document.getElementById('gps-info').innerText = '❌ GPS 미지원 브라우저';
}}
</script>
"""


# ============================================================
# 메인 UI
# ============================================================
st.title("🏃‍♂️ AI-Pacer")
tab_setup, tab_running, tab_result = st.tabs(["⚙️ 경로 설정", "🏃 러닝 중", "📊 결과 기록"])

# ============================================================
# TAB 1: 경로 설정
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
        rd      = st.session_state.route_data
        dist_km = rd["dist_m"] / 1000
        est_min = dist_km * st.session_state.target_pace

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("총 거리", f"{dist_km:.2f} km")
        col_b.metric("예상 완주", f"{int(est_min)}분 {int((est_min%1)*60)}초")
        temp, hum, w_desc = get_weather(st.session_state.s_lat, st.session_state.s_lng)
        col_c.metric("날씨", f"{temp}°C / {hum}%", delta=w_desc, delta_color="off")

        path = rd["path"]
        m = folium.Map(
            location=[(st.session_state.s_lat+st.session_state.e_lat)/2,
                      (st.session_state.s_lng+st.session_state.e_lng)/2],
            zoom_start=15)
        folium.PolyLine(path, color="#1E88E5", weight=6, opacity=0.9).add_to(m)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                      tooltip=f"출발: {st.session_state.s_name}",
                      icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                      tooltip=f"도착: {st.session_state.e_name}",
                      icon=folium.Icon(color="red", icon="flag")).add_to(m)
        st_folium(m, width=None, height=420)

        with st.expander("🧭 전체 경로 안내 보기"):
            for i, s in enumerate(rd["steps"]):
                dist_str = f"  ({int(s['distance'])}m)" if s['distance'] else ""
                st.write(f"**{i+1}.** {s['description']}{dist_str}")

        if st.button("🏃 러닝 시작!", type="primary", use_container_width=True):
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

    rd    = st.session_state.route_data
    steps = rd["steps"] if rd else []

    st.subheader("📡 실시간 자동 안내")
    st.caption("🔊 폰 음소거 해제 필수")
    components.html(
        build_auto_component(steps, st.session_state.target_pace),
        height=130
    )

    st.markdown("---")
    st.subheader("⚡ 페이스 대시보드")
    target    = st.session_state.target_pace
    current   = st.session_state.current_pace or target
    pace_diff = current - target
    elapsed   = int((datetime.now() - st.session_state.start_time).total_seconds()) \
                if st.session_state.start_time else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 목표", f"{target:.1f} min/km")
    col2.metric("🏃 현재", f"{current:.1f} min/km",
                delta=f"{'느림▲' if pace_diff>0 else '빠름▼'} {abs(pace_diff):.1f}",
                delta_color="inverse")
    col3.metric("⏱️ 경과", f"{elapsed//60}분 {elapsed%60}초")

    st.markdown("---")
    st.subheader("🗺️ 실시간 위치 지도")
    if rd:
        path    = rd["path"]
        cur_lat = st.session_state.current_lat or st.session_state.s_lat
        cur_lng = st.session_state.current_lng or st.session_state.s_lng

        m2 = folium.Map(location=[cur_lat, cur_lng], zoom_start=17)
        folium.PolyLine(path, color="#90CAF9", weight=5, opacity=0.6).add_to(m2)
        track = st.session_state.gps_track
        if len(track) >= 2:
            folium.PolyLine([[t[0], t[1]] for t in track],
                            color="#FF6D00", weight=5, opacity=0.95).add_to(m2)
        folium.CircleMarker([cur_lat, cur_lng], radius=10,
                            color="#1565C0", fill=True, fill_color="#1E88E5",
                            fill_opacity=0.9, tooltip="현재 위치").add_to(m2)
        folium.Marker([st.session_state.s_lat, st.session_state.s_lng],
                      icon=folium.Icon(color="green", icon="play")).add_to(m2)
        folium.Marker([st.session_state.e_lat, st.session_state.e_lng],
                      icon=folium.Icon(color="red", icon="flag")).add_to(m2)
        st_folium(m2, width=None, height=430)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔊 AI 상세 코칭", type="primary", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                t_lat = st.session_state.current_lat or st.session_state.s_lat
                t_lng = st.session_state.current_lng or st.session_state.s_lng
                temp, hum, w_desc = get_weather(t_lat, t_lng)
                pace_status = "느리게" if pace_diff > 0 else "빠르게"
                cur_step = st.session_state.current_nav_step
                nav_desc = steps[min(cur_step, len(steps)-1)]["description"] if steps else "직진"
                prompt = f"""
당신은 AI-Pacer 러닝 전문 코치입니다.
- 현재 내비: {nav_desc}
- 날씨: {temp}°C, 습도 {hum}%, {w_desc}
- 페이스: 목표 {target}분/km, 현재 {current}분/km (목표보다 {abs(pace_diff):.1f}분/km {pace_status})
- 경과: {elapsed//60}분 {elapsed%60}초
3문장 구어체: 페이스코칭 → 신체효율 팁 → 날씨 고려 조언
                """
                try:
                    resp  = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}])
                    msg   = resp.choices[0].message.content
                    audio = client.audio.speech.create(
                        model="tts-1", voice="alloy", input=msg)
                    audio.stream_to_file("guide.mp3")
                    st.success(f"🤖 {msg}")
                    st.audio("guide.mp3")
                except Exception as e:
                    st.error(f"OpenAI 오류: {e}")
    with col_b:
        if st.button("🛑 러닝 종료", type="secondary", use_container_width=True):
            st.session_state.running  = False
            st.session_state.finished = True
            st.rerun()

# ============================================================
# TAB 3: 결과 기록
# ============================================================
with tab_result:
    if not st.session_state.finished and not st.session_state.gps_track:
        st.info("러닝을 완료하면 여기에 결과가 기록됩니다.")
        st.stop()

    st.subheader("🏅 러닝 완료 리포트")
    track     = st.session_state.gps_track
    pace_hist = st.session_state.pace_history

    if track and len(track) >= 2:
        total_dist    = sum(calc_distance_m(track[i][0], track[i][1],
                                            track[i+1][0], track[i+1][1])
                            for i in range(len(track)-1)) / 1000
        elapsed_total = track[-1][2] - track[0][2]
        avg_pace      = (elapsed_total/60)/total_dist if total_dist > 0 else 0
        tgt           = st.session_state.target_pace

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏃 총 거리",     f"{total_dist:.2f} km")
        col2.metric("⏱️ 총 시간",    f"{int(elapsed_total//60)}분 {int(elapsed_total%60)}초")
        col3.metric("📈 평균 페이스", f"{avg_pace:.1f} min/km")
        col4.metric("🎯 목표 대비",   f"{avg_pace-tgt:+.1f} min/km", delta_color="inverse")

    st.subheader("🗺️ 러닝 발자취 지도")
    rd    = st.session_state.route_data
    s_lat = st.session_state.s_lat
    if rd and s_lat:
        path       = rd["path"]
        center_lat = (track[0][0]+track[-1][0])/2 if track else s_lat
        center_lng = (track[0][1]+track[-1][1])/2 if track else st.session_state.s_lng
        m3 = folium.Map(location=[center_lat, center_lng], zoom_start=15)
        folium.PolyLine(path, color="#9E9E9E", weight=3, opacity=0.5).add_to(m3)
        if len(track) >= 2:
            for i in range(len(track)-1):
                ratio = i / (len(track)-1)
                r, g  = int(255*ratio), int(255*(1-ratio))
                folium.PolyLine([[track[i][0],   track[i][1]],
                                 [track[i+1][0], track[i+1][1]]],
                                color=f"#{r:02x}{g:02x}00", weight=5).add_to(m3)
        if track:
            folium.Marker([track[0][0],  track[0][1]],
                          icon=folium.Icon(color="green", icon="play")).add_to(m3)
            folium.Marker([track[-1][0], track[-1][1]],
                          icon=folium.Icon(color="red", icon="stop")).add_to(m3)
        st_folium(m3, width=None, height=500)

    if pace_hist:
        st.subheader("📈 페이스 변화 그래프")
        df = pd.DataFrame({
            "페이스 (min/km)": pace_hist,
            "목표":            [st.session_state.target_pace]*len(pace_hist)
        })
        st.line_chart(df)

    if st.button("💾 기록 저장", use_container_width=True):
        record = {
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "route":        f"{st.session_state.s_name} → {st.session_state.e_name}",
            "track":        [(t[0], t[1]) for t in track],
            "pace_history": pace_hist,
            "target_pace":  st.session_state.target_pace,
        }
        st.download_button(
            "📥 JSON 다운로드",
            data=json.dumps(record, ensure_ascii=False, indent=2),
            file_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
