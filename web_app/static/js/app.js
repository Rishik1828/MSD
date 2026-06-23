/**
 * app.js — MUSICBEATS
 *
 * Screens: home → scan → prefs → songs
 * Key changes vs previous version:
 *  - SCAN_INTERVAL_MS raised to 1100ms (less lag, smoother)
 *  - Lower JPEG quality (0.65) → smaller payload → faster round-trip
 *  - Song list renders as horizontal rows (not cards) with visible titles
 *  - Genre filter uses pill buttons loaded from /api/filters (not a select)
 *  - Row equalizer bars animate when playing, pause when paused/stopped
 */

"use strict";

// Click-to-capture configuration

const EMOTION_META = {
  Happy:    { emoji: "😄", label: "Happy"     },
  Sad:      { emoji: "😢", label: "Sad"       },
  Anger:    { emoji: "😠", label: "Angry"     },
  Neutral:  { emoji: "😐", label: "Neutral"   },
  Fear:     { emoji: "😨", label: "Fearful"   },
  Contempt: { emoji: "😒", label: "Contempt"  },
  Disgust:  { emoji: "🤢", label: "Disgusted" },
  Surprise: { emoji: "😲", label: "Surprised" },
};

const LANG_EMOJI = { Telugu: "🎵", Hindi: "🎶", English: "🎸" };

// ── Global state ────────────────────────────────────────────────────────────
const State = {
  emotion:    null,
  emotionTag: null,
  filters:    { language: "", era: "", genre: "" },
  queue:      [],
  queueIdx:   -1,
  isPlaying:  false,
};


/* ══════════════════════════════════════════════════════════════════════════
   NAV — screen transitions
   ══════════════════════════════════════════════════════════════════════════ */
const Nav = (() => {
  let current = "home";

  function go(to, dir = "forward") {
    if (to === current) return;

    const fromEl = document.getElementById(`screen-${current}`);
    const toEl   = document.getElementById(`screen-${to}`);

    fromEl.classList.add(dir === "forward" ? "exit-left" : "exit-right");

    // Position new screen off-screen, then transition to center
    toEl.style.transform = dir === "forward" ? "translateX(28px)" : "translateX(-28px)";
    toEl.classList.add("active");
    toEl.getBoundingClientRect();  // force reflow
    toEl.style.transform = "";

    setTimeout(() => fromEl.classList.remove("active", "exit-left", "exit-right"), 340);
    current = to;
  }

  return { go };
})();


/* ══════════════════════════════════════════════════════════════════════════
   WEBCAM
   ══════════════════════════════════════════════════════════════════════════ */
const Webcam = (() => {
  let stream = null;
  const video  = document.getElementById("scanVideo");
  const canvas = document.getElementById("scanCanvas");
  const ctx    = canvas.getContext("2d");

  async function start() {
    if (stream) return true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      return true;
    } catch (e) {
      console.warn("Camera start failed:", e);
      return false;
    }
  }

  function stop() {
    if (!stream) return;
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
  }

  /** Capture frame as base64 JPEG. Unmirrored (correct for server-side CV). */
  function captureFrame() {
    if (!stream || video.readyState < 2) return null;
    const w = video.videoWidth || 640, h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -w, 0, w, h);
    ctx.restore();
    return canvas.toDataURL("image/jpeg", 0.65);  // lower quality = faster transfer
  }

  return { start, stop, captureFrame };
})();


/* ══════════════════════════════════════════════════════════════════════════
   SCAN — Click-to-capture, manual verification flow
   ══════════════════════════════════════════════════════════════════════════ */
const Scan = (() => {
  const topLabel    = document.getElementById("scanTopLabel");
  const emojiEl     = document.getElementById("scanEmoji");
  const emotionEl   = document.getElementById("scanEmotion");
  const confEl      = document.getElementById("scanConf");

  const video       = document.getElementById("scanVideo");
  const snapDisplay = document.getElementById("snapDisplay");
  const canvas      = document.getElementById("scanCanvas");
  const ctx         = canvas.getContext("2d");

  // State panels
  const stateIdle    = document.getElementById("scanStateIdle");
  const stateLoading = document.getElementById("scanStateLoading");
  const stateResult  = document.getElementById("scanStateResult");
  const stateError   = document.getElementById("scanStateError");
  const errorMsgEl   = document.getElementById("scanErrorMsg");

  let detecting     = false;
  let transitioning = false;
  let autoTimer     = null;

  function showState(state) {
    stateIdle.style.display    = state === "idle"    ? "block" : "none";
    stateLoading.style.display = state === "loading" ? "block" : "none";
    stateResult.style.display  = state === "result"  ? "block" : "none";
    stateError.style.display   = state === "error"   ? "block" : "none";
  }

  function start(cameraOk = true) {
    transitioning = false;
    detecting = false;
    if (autoTimer) clearTimeout(autoTimer);
    
    // Reset preview states
    snapDisplay.style.display = "none";
    clearBox();
    
    if (!cameraOk) {
      showCameraError();
      return;
    }
    
    // Show idle state (contains auto-scan loading animation)
    showState("idle");
    topLabel.innerHTML = `<span class="scan-live-dot"></span>Preparing camera...`;

    // Wait 1.5s for camera to auto-expose, then trigger auto-capture
    autoTimer = setTimeout(() => {
      if (!transitioning && !detecting) {
        capture();
      }
    }, 1500);
  }

  function stop() {
    transitioning = false;
    detecting = false;
    if (autoTimer) clearTimeout(autoTimer);
    clearBox();
    snapDisplay.style.display = "none";
  }

  async function capture() {
    if (detecting || transitioning) return;
    detecting = true;

    // Show loading state
    showState("loading");
    topLabel.innerHTML = `<span class="scan-live-dot" style="background:#8B7AEE"></span>Analyzing frame…`;

    // Freeze frame by drawing video to snapDisplay
    if (video.readyState >= 2) {
      snapDisplay.width = video.videoWidth || 640;
      snapDisplay.height = video.videoHeight || 480;
      const snapCtx = snapDisplay.getContext("2d");
      snapCtx.drawImage(video, 0, 0, snapDisplay.width, snapDisplay.height);
      snapDisplay.style.display = "block";
    }

    const frame = Webcam.captureFrame();
    if (!frame) {
      showError("Could not capture video frame.");
      detecting = false;
      return;
    }

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame }),
      });
      const data = await res.json();

      if (data.emotion && data.face_box) {
        // Save emotion to State
        State.emotion = data.emotion;
        State.emotionTag = data.emotion_tag;
        
        // Directly move to the next page (Prefs/Categories)
        proceed();
      } else {
        showError(data.message || "Move closer or improve lighting.");
      }
    } catch (e) {
      console.error("Capture Error:", e);
      showError("Connection or server error.");
    } finally {
      detecting = false;
    }
  }

  async function retry() {
    if (autoTimer) clearTimeout(autoTimer);
    const errTitle = document.querySelector("#scanStateError .scan-emotion");
    if (errTitle && errTitle.textContent === "Camera Access Denied") {
      const ok = await Webcam.start();
      if (ok) {
        if (errTitle) errTitle.textContent = "No face detected";
        const retryBtn = document.getElementById("retryBtn2");
        if (retryBtn) retryBtn.textContent = "Try Again";
        start(true);
      } else {
        window.location.reload();
      }
      return;
    }
    
    // Reset to live camera and trigger auto-capture again
    snapDisplay.style.display = "none";
    clearBox();
    showState("idle");
    topLabel.innerHTML = `<span class="scan-live-dot"></span>Preparing camera...`;
    
    autoTimer = setTimeout(() => {
      if (!transitioning && !detecting) {
        capture();
      }
    }, 1500);
  }

  function proceed() {
    if (!State.emotion) return;
    transitioning = true;
    stop();
    Webcam.stop();
    Prefs.setMood(State.emotion);
    Nav.go("prefs", "forward");
  }

  function showCameraError() {
    clearBox();
    showState("error");
    const errTitle = document.querySelector("#scanStateError .scan-emotion");
    if (errTitle) errTitle.textContent = "Camera Access Denied";
    errorMsgEl.textContent = "Please allow camera access in your browser's address bar and reload.";
    const retryBtn = document.getElementById("retryBtn2");
    if (retryBtn) retryBtn.textContent = "Reload Page to Retry";
    topLabel.innerHTML = `<span class="scan-live-dot" style="background:#EF4444;animation:none"></span>Camera Blocked`;
  }

  function showError(msg) {
    const errTitle = document.querySelector("#scanStateError .scan-emotion");
    if (errTitle) errTitle.textContent = "No face detected";
    const retryBtn = document.getElementById("retryBtn2");
    if (retryBtn) retryBtn.textContent = "Try Again";

    errorMsgEl.textContent = msg;
    clearBox();
    showState("error");
    topLabel.innerHTML = `<span class="scan-live-dot" style="background:#EF4444;animation:none"></span>No face detected`;
  }

  // ── Face box: 4 corner L-brackets ─────────────────────────────────────
  function drawBox(box) {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    canvas.width = vw;
    canvas.height = vh;
    ctx.clearRect(0, 0, vw, vh);

    const x = box.x * vw;
    const y = box.y * vh;
    const w = box.w * vw;
    const h = box.h * vh;
    const cl = Math.min(w, h) * 0.2;

    ctx.strokeStyle = "#22C55E";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";

    const corners = [
      [[x, y + cl], [x, y], [x + cl, y]],
      [[x + w - cl, y], [x + w, y], [x + w, y + cl]],
      [[x, y + h - cl], [x, y + h], [x + cl, y + h]],
      [[x + w - cl, y + h], [x + w, y + h], [x + w, y + h - cl]],
    ];
    corners.forEach(([a, b, c]) => {
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.stroke();
    });

    ctx.fillStyle = "rgba(34,197,94,0.06)";
    ctx.fillRect(x, y, w, h);
  }

  function clearBox() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  return { start, stop, capture, retry, proceed };
})();


/* ══════════════════════════════════════════════════════════════════════════
   PREFS — pill groups, genre pills from API, "Find Songs"
   ══════════════════════════════════════════════════════════════════════════ */
const Prefs = (() => {
  const moodEmojiEl = document.getElementById("prefsMoodEmoji");
  const moodLabelEl = document.getElementById("prefsMoodLabel");
  const findBtn     = document.getElementById("findSongsBtn");

  function setMood(emotion) {
    const meta = EMOTION_META[emotion] || { emoji: "🎭", label: emotion };
    moodEmojiEl.textContent = meta.emoji;
    moodLabelEl.textContent = meta.label;
    document.getElementById("songsHeaderEmoji").textContent = meta.emoji;
    document.getElementById("songsHeaderLabel").textContent = `${meta.label} mood`;
  }

  /** Populate genre pill buttons from /api/filters (dataset-only genres) */
  async function loadGenres() {
    const container = document.getElementById("genrePills");
    try {
      const res  = await fetch("/api/filters");
      const data = await res.json();

      (data.genres || []).forEach(g => {
        const btn = document.createElement("button");
        btn.className = "pill"; btn.type = "button";
        btn.dataset.value = g; btn.textContent = g;
        container.appendChild(btn);
      });

      // Wire up all pills in the genre group (including "All")
      initPills("genrePills", "genre");
    } catch (e) { console.warn("Genre load failed:", e); }
  }

  /** Single-select pill group — clicking a pill marks it active, saves to State.filters */
  function initPills(groupId, stateKey) {
    const group = document.getElementById(groupId);
    group.querySelectorAll(".pill").forEach(btn => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        State.filters[stateKey] = btn.dataset.value;
      });
    });
  }

  findBtn.addEventListener("click", () => {
    Nav.go("songs", "forward");
    Songs.fetch(true);
  });

  return { setMood, loadGenres, initPills };
})();


/* ══════════════════════════════════════════════════════════════════════════
   SONGS — fetch, render as list rows
   ══════════════════════════════════════════════════════════════════════════ */
const Songs = (() => {
  const listEl  = document.getElementById("songList");
  const countEl = document.getElementById("songsCount");

  async function fetch(autoPlay = false) {
    showSkeletons();

    const p = new URLSearchParams();
    if (State.emotionTag)        p.set("emotion",  State.emotionTag);
    if (State.filters.language)  p.set("language", State.filters.language);
    if (State.filters.era)       p.set("era",      State.filters.era);
    if (State.filters.genre)     p.set("genre",    State.filters.genre);

    try {
      const res  = await window.fetch(`/api/songs?${p}`);
      const data = await res.json();

      if (data.error) { showError(data.hint || data.error); return; }

      State.queue    = data.songs;
      State.queueIdx = -1;
      render(data.songs);
      countEl.textContent = `${data.count} track${data.count !== 1 ? "s" : ""}`;

      if (autoPlay) {
        const first = data.songs.findIndex(s => s.has_file);
        if (first >= 0) setTimeout(() => Player.playIndex(first), 280);
      }
    } catch (e) {
      showError("Could not reach server.");
    }
  }

  function render(songs) {
    if (!songs.length) {
      listEl.innerHTML = `
        <div class="list-state">
          <div class="list-state__icon">🎵</div>
          <div class="list-state__title">No tracks found</div>
          <div class="list-state__sub">Try clearing some filters.</div>
        </div>`;
      return;
    }

    listEl.innerHTML = songs.map((s, i) => {
      const canPlay = s.has_file;
      const emoji   = LANG_EMOJI[s.language] || "🎵";
      return `
        <div class="song-row${canPlay ? "" : " no-file"}"
             data-index="${i}"
             role="button" tabindex="${canPlay ? 0 : -1}"
             aria-label="${canPlay ? `Play ${esc(s.title)}` : `${esc(s.title)} — no audio`}">

          <div class="song-row__num">${i + 1}</div>
          <div class="song-row__play-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="currentColor" width="14"><polygon points="5,3 19,12 5,21"/></svg>
          </div>

          <div class="song-row__art" aria-hidden="true">${emoji}</div>

          <div class="song-row__info">
            <div class="song-row__title">${esc(s.title)}</div>
            <div class="song-row__artist">${esc(s.artist)}${s.era ? " · " + esc(s.era) : ""}</div>
          </div>

          <div class="song-row__tags">
            <span class="badge ${s.language}">${esc(s.language)}</span>
            ${s.genre ? `<span class="badge">${esc(s.genre)}</span>` : ""}
          </div>

          <div class="song-row__eq paused" aria-hidden="true">
            <div class="eq-bar"></div><div class="eq-bar"></div>
            <div class="eq-bar"></div><div class="eq-bar"></div>
            <div class="eq-bar"></div>
          </div>
        </div>`;
    }).join("");

    listEl.querySelectorAll(".song-row:not(.no-file)").forEach(el => {
      const handler = () => Player.playIndex(+el.dataset.index);
      el.addEventListener("click", handler);
      el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") handler(); });
    });
  }

  function setActiveRow(idx) {
    listEl.querySelectorAll(".song-row").forEach((el, i) => {
      const active = i === idx;
      el.classList.toggle("active", active);
      const eq = el.querySelector(".song-row__eq");
      if (eq) eq.classList.toggle("paused", !active || !State.isPlaying);
    });

    const activeEl = listEl.querySelector(".song-row.active");
    if (activeEl) activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /** Sync active row eq animation with play/pause state */
  function syncEqAnimation() {
    const eq = listEl.querySelector(".song-row.active .song-row__eq");
    if (eq) eq.classList.toggle("paused", !State.isPlaying);
  }

  function showSkeletons() {
    listEl.innerHTML = Array.from({ length: 8 }, (_, i) => `
      <div class="song-row-skeleton">
        <div class="skel" style="width:22px;height:12px;border-radius:3px"></div>
        <div class="skel" style="width:44px;height:44px;border-radius:10px;flex-shrink:0"></div>
        <div style="flex:1;display:flex;flex-direction:column;gap:6px">
          <div class="skel" style="width:${50+Math.random()*25|0}%;height:13px"></div>
          <div class="skel" style="width:${30+Math.random()*20|0}%;height:11px"></div>
        </div>
        <div class="skel" style="width:52px;height:20px;border-radius:5px"></div>
      </div>`).join("");
    countEl.textContent = "";
  }

  function showError(msg) {
    listEl.innerHTML = `
      <div class="list-state">
        <div class="list-state__icon">⚠️</div>
        <div class="list-state__title">Could not load songs</div>
        <div class="list-state__sub">${esc(msg)}</div>
      </div>`;
  }

  return { fetch, setActiveRow, syncEqAnimation };
})();


/* ══════════════════════════════════════════════════════════════════════════
   PLAYER — HTMLAudioElement transport
   ══════════════════════════════════════════════════════════════════════════ */
const Player = (() => {
  const audio    = document.getElementById("audioEl");
  const ppBtn    = document.getElementById("npPlayPause");
  const prevBtn  = document.getElementById("npPrev");
  const nextBtn  = document.getElementById("npNext");
  const stopBtn  = document.getElementById("npStop");
  const seekEl   = document.getElementById("npSeek");
  const volEl    = document.getElementById("npVol");
  const curEl    = document.getElementById("npCurTime");
  const durEl    = document.getElementById("npDuration");
  const titleEl  = document.getElementById("npTitle");
  const artistEl = document.getElementById("npArtist");
  const artEl    = document.getElementById("npArt");
  const iconPlay = document.getElementById("iconPlay");
  const iconPause= document.getElementById("iconPause");
  const npEq     = document.getElementById("equalizer");

  ppBtn.addEventListener("click",  () => audio.src ? (audio.paused ? audio.play() : audio.pause()) : playIndex(0));
  prevBtn.addEventListener("click",() => playIndex(State.queueIdx - 1));
  nextBtn.addEventListener("click",() => playIndex(State.queueIdx + 1));
  stopBtn.addEventListener("click", stopAll);

  seekEl.addEventListener("input", () => { if (audio.duration) audio.currentTime = seekEl.value; });
  volEl.addEventListener("input",  () => { audio.volume = volEl.value / 100; });

  audio.addEventListener("timeupdate", () => {
    if (!audio.duration) return;
    seekEl.value = audio.currentTime;
    const pct = (audio.currentTime / audio.duration) * 100;
    seekEl.style.background = `linear-gradient(to right, #5E4AE3 ${pct}%, rgba(255,255,255,.2) 0%)`;
    curEl.textContent = fmt(audio.currentTime);
  });

  audio.addEventListener("loadedmetadata", () => {
    seekEl.max = Math.floor(audio.duration);
    durEl.textContent = fmt(audio.duration);
  });

  audio.addEventListener("ended",  () => playIndex(State.queueIdx + 1));
  audio.addEventListener("play",   () => setPlayState(true));
  audio.addEventListener("pause",  () => setPlayState(false));

  function playIndex(idx) {
    if (idx < 0 || idx >= State.queue.length) return;
    // Skip songs without a file
    let target = idx;
    while (target < State.queue.length && !State.queue[target].has_file) target++;
    if (target >= State.queue.length) return;

    State.queueIdx = target;
    const s = State.queue[target];

    audio.src = `/api/audio/${s.song_id}`;
    audio.load();
    audio.volume = volEl.value / 100;
    audio.play().catch(() => {});

    titleEl.textContent  = s.title;
    artistEl.textContent = `${s.artist} · ${s.language}`;
    artEl.textContent    = LANG_EMOJI[s.language] || "♪";

    seekEl.value = 0; seekEl.style.background = "";
    curEl.textContent = "0:00"; durEl.textContent = "0:00";

    Songs.setActiveRow(target);
  }

  function stopAll() {
    audio.pause(); audio.currentTime = 0; audio.src = "";
    seekEl.value = 0; seekEl.style.background = "";
    curEl.textContent = "0:00"; durEl.textContent = "0:00";
    titleEl.textContent = "No track selected";
    artistEl.textContent = "Pick a song above to play";
    artEl.textContent = "♪";
    State.queueIdx = -1;
    Songs.setActiveRow(-1);
    setPlayState(false);
  }

  function setPlayState(playing) {
    State.isPlaying = playing;
    iconPlay.style.display  = playing ? "none"   : "inline";
    iconPause.style.display = playing ? "inline" : "none";
    npEq.classList.toggle("paused", !playing);
    Songs.syncEqAnimation();
  }

  return { playIndex };
})();


/* ══════════════════════════════════════════════════════════════════════════
   HELPERS
   ══════════════════════════════════════════════════════════════════════════ */
function esc(s) {
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function fmt(s) {
  if (!isFinite(s)) return "0:00";
  return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`;
}


/* ══════════════════════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {

  // Prefs: static pill groups
  Prefs.initPills("langPills", "language");
  Prefs.initPills("eraPills",  "era");
  // Genre pills loaded from API (dataset-only genres)
  Prefs.loadGenres();

  // Home → Scan
  document.getElementById("startBtn").addEventListener("click", async () => {
    Nav.go("scan", "forward");
    const ok = await Webcam.start();
    Scan.start(ok);
  });

  // Scan interaction buttons
  document.getElementById("retryBtn").addEventListener("click", () => Scan.retry());
  document.getElementById("retryBtn2").addEventListener("click", () => Scan.retry());
  document.getElementById("continueBtn").addEventListener("click", () => Scan.proceed());

  // Scan → Home
  document.getElementById("scanBackBtn").addEventListener("click", () => {
    Scan.stop(); Webcam.stop(); Nav.go("home", "backward");
  });

  // Prefs → Scan
  document.getElementById("prefsBackBtn").addEventListener("click", async () => {
    Nav.go("scan", "backward");
    const ok = await Webcam.start();
    Scan.start(ok);
  });

  // Songs → Prefs
  document.getElementById("songsBackBtn").addEventListener("click", () => {
    Nav.go("prefs", "backward");
  });

});
