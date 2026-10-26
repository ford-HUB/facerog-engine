"use strict";

const API = "/api/v1";
const TRACK_POLL_MS = 250;
const EVENTS_POLL_MS = 1000;
const HEALTH_POLL_MS = 3000;
const SERIES_SLOTS = 8;
const STATUS_LABELS = { matched: "✓ Recognized", unknown: "Not registered", identifying: "Identifying…" };

const state = {
  tracks: [],
  matchThreshold: 0.4,
  personColorSlots: new Map(),
  cards: new Map(),
};

const $ = (id) => document.getElementById(id);

function trackColorVar(track) {
  if (track.status !== "matched" || !track.person) return "--status-neutral";
  const id = track.person.person_id;
  if (!state.personColorSlots.has(id) && state.personColorSlots.size < SERIES_SLOTS) {
    state.personColorSlots.set(id, state.personColorSlots.size + 1);
  }
  const slot = state.personColorSlots.get(id);
  return slot ? `--series-${slot}` : "--status-neutral";
}

function trackLabel(track) {
  if (track.person) return track.person.name;
  return track.status === "identifying" ? "Identifying…" : "Unknown";
}

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

async function refreshHealth() {
  try {
    const health = await fetchJson(`${API}/health`);
    const streamPill = $("stream-pill");
    const live = health.stream_running && health.video_source_open;
    streamPill.textContent = live ? "Stream: live" : health.stream_running ? "Stream: no camera" : "Stream: stopped";
    streamPill.className = `pill ${live ? "good" : "bad"}`;
    $("enrolled-pill").textContent = `${health.enrolled_people} enrolled`;
    const gpu = health.execution_providers.find((p) => !p.startsWith("CPU"));
    $("provider-pill").textContent = gpu ? gpu.replace("ExecutionProvider", " GPU") : "CPU";
  } catch (error) {
    $("stream-pill").textContent = "Engine offline";
    $("stream-pill").className = "pill bad";
  }
}

function connectVideo() {
  const video = $("video");
  video.onload = () => { $("video-empty").hidden = true; };
  video.onerror = () => {
    $("video-empty").hidden = false;
    setTimeout(connectVideo, 2000);
  };
  video.src = `${API}/stream/video?t=${Date.now()}`;
}

async function refreshTracks() {
  try {
    const snapshot = await fetchJson(`${API}/stream/tracks`);
    state.tracks = snapshot.tracks;
    state.matchThreshold = snapshot.match_threshold;
    $("fps-pill").textContent = `${snapshot.fps.toFixed(1)} fps`;
  } catch {
    state.tracks = [];
  }
  renderTrackCards();
}

function createTrackCard() {
  const card = document.createElement("article");
  card.className = "track-card";
  card.innerHTML = `
    <header>
      <span class="name"></span>
      <span class="chips"></span>
    </header>
    <div class="faces-compare">
      <figure><div class="registered"></div><figcaption>Registered photo</figcaption></figure>
      <figure><img class="zoom" alt="Zoomed live face with landmark structure"><figcaption>Live zoom · face structure</figcaption></figure>
    </div>
    <div class="metric-row"><span>Similarity</span><strong class="similarity">—</strong></div>
    <div class="similarity-bar"><div class="similarity-fill"></div><div class="threshold-mark" title="Match threshold"></div></div>
    <div class="metric-row"><span class="pose"></span><span class="meta"></span></div>`;
  return card;
}

function renderTrackCards() {
  const container = $("track-cards");
  const activeIds = new Set(state.tracks.map((track) => track.track_id));
  for (const [id, card] of state.cards) {
    if (!activeIds.has(id)) { card.remove(); state.cards.delete(id); }
  }
  container.querySelector(".empty")?.remove();
  if (!state.tracks.length) {
    container.innerHTML = '<p class="empty">No faces in view.</p>';
    return;
  }
  for (const track of state.tracks) {
    let card = state.cards.get(track.track_id);
    if (!card) {
      card = createTrackCard();
      state.cards.set(track.track_id, card);
      container.append(card);
    }
    updateTrackCard(card, track);
  }
}

function updateTrackCard(card, track) {
  card.style.setProperty("--track-color", `var(${trackColorVar(track)})`);
  card.querySelector(".name").textContent = `#${track.track_id} ${trackLabel(track)}`;
  const chips = [`<span class="chip ${track.status}">${STATUS_LABELS[track.status]}</span>`];
  if (track.head_pose.is_angled) chips.push('<span class="chip angled">angled · zoomed</span>');
  card.querySelector(".chips").innerHTML = chips.join(" ");

  const registered = card.querySelector(".registered");
  const photoUrl = track.person ? track.person.photo_url : null;
  if (registered.dataset.url !== String(photoUrl)) {
    registered.dataset.url = String(photoUrl);
    registered.innerHTML = photoUrl
      ? `<img src="${photoUrl}" alt="Registered photo of ${escapeHtml(track.person.name)}">`
      : `<div class="placeholder">${track.status === "identifying" ? "Identifying…" : "No match"}</div>`;
  }
  card.querySelector(".zoom").src = `${track.zoom_url}?t=${Date.now()}`;

  const similarity = track.similarity;
  card.querySelector(".similarity").textContent =
    similarity === null ? "—" : `${similarity.toFixed(3)} ${similarity >= state.matchThreshold ? "≥" : "<"} ${state.matchThreshold.toFixed(2)}`;
  const toPercent = (value) => `${Math.max(0, Math.min(1, value)) * 100}%`;
  card.querySelector(".similarity-fill").style.width = toPercent(similarity ?? 0);
  card.querySelector(".threshold-mark").style.left = toPercent(state.matchThreshold);
  const pose = track.head_pose;
  card.querySelector(".pose").textContent =
    `yaw ${pose.yaw_degrees.toFixed(0)}° · pitch ${pose.pitch_degrees.toFixed(0)}° · roll ${pose.roll_degrees.toFixed(0)}°`;
  card.querySelector(".meta").textContent = `${track.seconds_tracked.toFixed(0)}s · ${track.recognition_count} checks`;
}

function escapeHtml(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span.innerHTML;
}

async function refreshPeople() {
  const grid = $("people-grid");
  try {
    const { faces } = await fetchJson(`${API}/faces`);
    grid.replaceChildren(...faces.map(renderPerson));
    if (!faces.length) grid.innerHTML = '<p class="empty">No one enrolled. Add photos to datasets/faces/&lt;name&gt;/ or use the form.</p>';
  } catch (error) {
    grid.innerHTML = `<p class="empty">Could not load people: ${escapeHtml(error.message)}</p>`;
  }
}

function renderPerson(face) {
  const card = document.createElement("div");
  card.className = "person";
  card.innerHTML = `
    <img src="${face.photo_url}" alt="Photo of ${escapeHtml(face.name)}" loading="lazy">
    <div class="info"><strong>${escapeHtml(face.name)}</strong><span>${face.image_count} photos · ${face.template_count} templates</span></div>
    <button type="button" class="secondary">Remove</button>`;
  card.querySelector("button").addEventListener("click", async () => {
    if (!confirm(`Remove ${face.name} and delete their dataset folder?`)) return;
    await fetchJson(`${API}/faces/${encodeURIComponent(face.person_id)}`, { method: "DELETE" });
    refreshPeople();
    refreshHealth();
  });
  return card;
}

$("enroll-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const message = $("enroll-message");
  message.className = "form-message";
  message.textContent = "Enrolling…";
  try {
    const face = await fetchJson(`${API}/faces`, { method: "POST", body: new FormData(form) });
    message.textContent = `Enrolled ${face.name} (${face.image_count} photos).`;
    form.reset();
    refreshPeople();
    refreshHealth();
  } catch (error) {
    message.className = "form-message error";
    message.textContent = error.message;
  }
});

$("reload-button").addEventListener("click", async () => {
  const message = $("enroll-message");
  message.className = "form-message";
  message.textContent = "Reloading dataset…";
  try {
    const summary = await fetchJson(`${API}/faces/reload`, { method: "POST" });
    const skipped = summary.skipped_images.length ? `, skipped ${summary.skipped_images.join(", ")}` : "";
    message.textContent = `Enrolled ${summary.person_count} people (${summary.template_count} templates)${skipped}.`;
  } catch (error) {
    message.className = "form-message error";
    message.textContent = error.message;
  }
  refreshPeople();
  refreshHealth();
});

async function refreshEvents() {
  try {
    const events = await fetchJson(`${API}/stream/events`);
    $("events").replaceChildren(...events.map((event) => {
      const item = document.createElement("li");
      const time = new Date(event.timestamp * 1000).toLocaleTimeString();
      item.innerHTML = `
        <img src="${event.person.photo_url}" alt="">
        <span><strong>${escapeHtml(event.person.name)}</strong> · track #${event.track_id} · ${event.similarity.toFixed(3)}</span>
        <span class="time">${time}</span>`;
      return item;
    }));
  } catch { /* the stream may be stopped; keep the last log */ }
}

connectVideo();
refreshHealth();
refreshPeople();
setInterval(refreshTracks, TRACK_POLL_MS);
setInterval(refreshEvents, EVENTS_POLL_MS);
setInterval(refreshHealth, HEALTH_POLL_MS);
