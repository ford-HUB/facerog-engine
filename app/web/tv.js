"use strict";

const API = "/api/v1";
const POLL_MS = 500;
const REPLAY_RECENT_SECONDS = 30;
const UPLOAD_NOTICE_MS = 5000;

const tv = {
  sessionId: null,
  lastSequence: 0,
  displaySeconds: 3,
  items: [],
  current: null,
  isPolling: false,
  isQueueVisible: false,
  isSidebarPinned: false,
};

const $ = (id) => document.getElementById(id);

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function formatTime(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function escapeHtml(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span.innerHTML;
}

async function pollCheckIns() {
  if (tv.isPolling) return;
  tv.isPolling = true;
  try {
    const data = await fetchJson(`${API}/check-ins?after_sequence=${tv.lastSequence}`);
    const isFirstLoad = tv.sessionId === null;
    if (data.session_id !== tv.sessionId) {
      tv.sessionId = data.session_id;
      tv.items = [];
      tv.current = null;
      tv.lastSequence = 0;
      if (!isFirstLoad) {
        tv.isPolling = false;
        return pollCheckIns();
      }
    }
    tv.displaySeconds = data.display_seconds;
    tv.lastSequence = data.last_sequence;
    // On first load, only check-ins from the last few seconds are replayed, so a refresh
    // doesn't replay the whole session but someone who just arrived is still shown.
    const replayAfter = Date.now() / 1000 - REPLAY_RECENT_SECONDS;
    for (const checkIn of data.check_ins) {
      const isOld = isFirstLoad && checkIn.checked_in_at < replayAfter;
      tv.items.push({ checkIn, state: isOld ? "done" : "waiting" });
    }
    if (data.check_ins.length || isFirstLoad) renderQueue();
    showNext();
  } catch {
    /* engine restarting; keep the current screen and retry */
  } finally {
    tv.isPolling = false;
  }
}

function showNext() {
  if (tv.current) return;
  const next = tv.items.find((item) => item.state === "waiting");
  if (!next) {
    renderIdle();
    return;
  }
  next.state = "showing";
  tv.current = next;
  renderSpotlight(next);
  renderQueue();
  setTimeout(() => {
    if (tv.current !== next) return;
    next.state = "done";
    tv.current = null;
    renderQueue();
    showNext();
  }, tv.displaySeconds * 1000);
}

function renderSpotlight(item) {
  const { person, checked_in_at: checkedInAt } = item.checkIn;
  $("idle").hidden = true;
  const spotlight = $("spotlight");
  spotlight.hidden = false;
  spotlight.style.animation = "none";
  void spotlight.offsetWidth;
  spotlight.style.animation = "";
  $("spotlight-photo").src = person.photo_url;
  $("spotlight-photo").alt = `Photo of ${person.name}`;
  $("spotlight-name").textContent = person.name;
  $("spotlight-time").textContent = `Checked in at ${formatTime(checkedInAt)}`;
  const waiting = tv.items.filter((i) => i.state === "waiting").length;
  $("spotlight-next").textContent = waiting ? `${waiting} more in queue` : "";
  const fill = $("progress-fill");
  fill.classList.remove("running");
  void fill.offsetWidth;
  fill.style.animationDuration = `${tv.displaySeconds}s`;
  fill.classList.add("running");
}

function renderIdle() {
  $("spotlight").hidden = true;
  $("idle").hidden = false;
}

const STATE_ORDER = { showing: 0, waiting: 1, done: 2 };

function renderQueue() {
  const ordered = [...tv.items].sort((a, b) => {
    if (a.state !== b.state) return STATE_ORDER[a.state] - STATE_ORDER[b.state];
    const direction = a.state === "done" ? -1 : 1;
    return direction * (a.checkIn.sequence - b.checkIn.sequence);
  });
  let linePosition = 0;
  $("queue").replaceChildren(...ordered.map((item) => {
    if (item.state === "waiting") linePosition += 1;
    return renderQueueItem(item, linePosition);
  }));
  $("queue-empty").hidden = tv.items.length > 0;
  $("count").textContent = String(tv.items.length);
  updateSidebarVisibility();
}

/** The queue only appears when two or more people are waiting together; one person gets the full screen. */
function updateSidebarVisibility() {
  const queued = tv.items.filter((item) => item.state !== "done").length;
  if (queued >= 2) tv.isQueueVisible = true;
  else if (queued === 0) tv.isQueueVisible = false;
  document.body.classList.toggle("sidebar-hidden", !tv.isQueueVisible && !tv.isSidebarPinned);
}

/** Where the person is in line. Everyone in the queue is already checked in; this only says when they are shown. */
function queuePositionLabel(item, linePosition) {
  if (item.state === "showing") return "Now showing";
  if (item.state === "done") return "Shown";
  return linePosition === 1 ? "Up next" : `#${linePosition} in line`;
}

function renderQueueItem(item, linePosition) {
  const { person, checked_in_at: checkedInAt, source } = item.checkIn;
  const sourceLabel = source === "photo" ? " · photo" : "";
  const element = document.createElement("li");
  element.className = `queue-item ${item.state}`;
  element.innerHTML = `
    <img src="${person.photo_url}" alt="" loading="lazy">
    <div>
      <div class="name">${escapeHtml(person.name)}</div>
      <div class="meta">
        <span class="chip checked">✓ Checked in</span>
        <span class="position">${queuePositionLabel(item, linePosition)}</span>
        <span>${formatTime(checkedInAt)}${sourceLabel}</span>
      </div>
    </div>
    <button type="button" class="remove-button" title="Remove check-in" aria-label="Remove check-in for ${escapeHtml(person.name)}">×</button>`;
  element.querySelector(".remove-button").addEventListener("click", () => removeCheckIn(item));
  return element;
}

async function removeCheckIn(item) {
  try {
    await fetchJson(`${API}/check-ins/${item.checkIn.sequence}`, { method: "DELETE" });
  } catch {
    /* already removed on the server; drop it locally too */
  }
  tv.items = tv.items.filter((other) => other !== item);
  if (tv.current === item) tv.current = null;
  renderQueue();
  showNext();
}

/** Check in everyone in the photo; they arrive through the normal poll and queue up together. */
async function uploadPhoto(file) {
  const body = new FormData();
  body.append("image", file);
  showUploadNotice("Checking in photo…");
  try {
    const result = await fetchJson(`${API}/check-ins`, { method: "POST", body });
    showUploadNotice(describePhotoResult(result));
    pollCheckIns();
  } catch (error) {
    showUploadNotice(error.message, true);
  }
}

function describePhotoResult(result) {
  const count = (status) => result.faces.filter((face) => face.status === status).length;
  const parts = [`${result.checked_in_count} added to the queue`];
  if (count("already_checked_in")) parts.push(`${count("already_checked_in")} already checked in`);
  if (count("not_registered")) parts.push(`${count("not_registered")} not registered`);
  return `${result.faces.length} faces found · ${parts.join(" · ")}`;
}

let uploadNoticeTimer = null;

function showUploadNotice(text, isError = false) {
  const notice = $("upload-notice");
  notice.textContent = text;
  notice.classList.toggle("error", isError);
  notice.hidden = false;
  clearTimeout(uploadNoticeTimer);
  uploadNoticeTimer = setTimeout(() => { notice.hidden = true; }, UPLOAD_NOTICE_MS);
}

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function connectCamera() {
  const preview = $("camera-preview");
  preview.onerror = () => setTimeout(connectCamera, 3000);
  preview.src = `${API}/stream/video?t=${Date.now()}`;
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
}

$("fullscreen-button").addEventListener("click", toggleFullscreen);
$("upload-button").addEventListener("click", () => $("photo-input").click());
$("photo-input").addEventListener("change", (event) => {
  const [file] = event.target.files;
  event.target.value = "";
  if (file) uploadPhoto(file);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "f" || event.key === "F") toggleFullscreen();
  if (event.key === "u" || event.key === "U") $("photo-input").click();
  if (event.key === "q" || event.key === "Q") {
    tv.isSidebarPinned = !tv.isSidebarPinned;
    updateSidebarVisibility();
  }
});

$("reset-button").addEventListener("click", async () => {
  if (!confirm("Start a new session? The queue is cleared and everyone can check in again.")) return;
  await fetchJson(`${API}/check-ins`, { method: "DELETE" }).catch(() => {});
  pollCheckIns();
});

tickClock();
setInterval(tickClock, 1000);
connectCamera();
pollCheckIns();
setInterval(pollCheckIns, POLL_MS);
