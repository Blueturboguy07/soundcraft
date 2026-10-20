//! AI through publik API or the person's own key, desktop only. The setup flow, the credential
//! file and the HTTP calls live in the shared `craft-publik` crate (one per Craft app). This module
//! is SoundCraft's thin host:
//!
//! - the setup window on the first launch (once, after a few frames, when no dialog, recovery offer
//!   or bounce shows; "Later" ends it for good; never while an agent drives the app over
//!   `--control`),
//! - Setup › Preferences… (its AI section: status, sign-in and key controls, the setup flow in
//!   place), also reached from Clip › AI › AI Settings…,
//! - Clip › AI › Transcribe to Markers… and Text to Speech Clip…, two windows.
//!
//! Nothing is sent, and nothing is charged, until the person clicks Transcribe or Generate.
//!
//! **Transcribe to Markers** renders the selected clip as Export Clips as Files does (clip gain,
//! fades, clip effects), or the session mix from the selection's start to its end as Bounce Mix
//! does (a slice per frame, on the UI thread, like a bounce: hosted plugins keep their thread
//! rules). The audio is mixed to mono, resampled to 16 kHz and cut into parts of at most a minute,
//! each cut at the quietest moment of the part's last seconds. SoundCraft has no lossy encoder, so
//! each part goes out as 16-bit FLAC (its own encoder): under 2 MB for a minute of any audio, well
//! inside the 4.5 MB one request to publik API can carry, and a minute's hold on a publik balance
//! stays near $0.03. Each segment that comes back becomes one marker at the segment's start, named
//! with its words, through the engine's own `markers.add` command, all in one undo step.
//!
//! **Text to Speech Clip** sends the text and the chosen voice through `client::speech`. The audio
//! file goes where recordings go (`Audio Files` next to the session, else the session's autosave
//! folder) and onto the timeline through the engine's own `file.import_audio` command: a new clip
//! at the edit cursor on the first free audio track of the same width (a new track when every one
//! is busy), one undo step.
//!
//! A paid result goes in when it arrives. When it cannot (the session it was made for was closed,
//! playback or recording runs, a command fails), it waits behind Place Markers / Place Clip.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use craft_publik::client::{self, Fail, Reply};
use craft_publik::ui::{Action, AiState, spawn};
use craft_publik::{Line, Provider, WHY_IT_COSTS};
use egui::RichText;
use serde_json::{Value, json};
use soundcraft_engine::TransportRequest;
use soundcraft_model::{ClipId, Session, TrackId, TrackKind};
use soundcraft_time::{Range, Samples};

pub use craft_publik::AppInfo;

use crate::SoundApp;
use crate::theme::{Tokens, bold};

/// Clip › AI › Transcribe to Markers… (a window; nothing is sent from the menu).
pub const TRANSCRIBE: &str = "window.ai_transcribe";
/// Clip › AI › Text to Speech Clip…
pub const SPEECH: &str = "window.ai_speech";
/// Setup › Preferences… and Clip › AI › AI Settings…: the Preferences window, its AI section.
pub const PREFERENCES: &str = "window.preferences";
pub const TRANSCRIBE_PATH: &str = "Clip > AI > Transcribe to Markers...";
pub const SPEECH_PATH: &str = "Clip > AI > Text to Speech Clip...";
pub const SETTINGS_PATH: &str = "Clip > AI > AI Settings...";
pub const PREFERENCES_PATH: &str = "Setup > Preferences...";

/// The audio a transcription sends: 16 kHz mono, 16-bit FLAC.
pub const SEND_RATE: u32 = 16_000;
/// The longest part one request sends. A minute of FLAC at 16 kHz is under 2 MB whatever the
/// audio, and publik holds about $0.03 of the balance for it while it runs.
const PART_SECONDS: usize = 60;
/// A part ends at the quietest moment of its last seconds, so a cut seldom splits a word.
const SEARCH_SECONDS: usize = 10;
/// A rest this short at the end goes in the last part instead of a part of its own.
const TAIL_SECONDS: usize = 2;
/// The longest audio one click sends (15 parts).
pub const MAX_TRANSCRIBE_SECONDS: f64 = 15.0 * 60.0;
/// The shortest audio the transcription line takes.
const MIN_TRANSCRIBE_SECONDS: f64 = 0.5;
/// The most markers one transcription makes.
const MAX_MARKERS: usize = 1_000;
/// The longest marker name, in characters.
const MAX_NAME_CHARS: usize = 255;
/// The longest text one speech clip takes (the speech line's limit), in characters.
pub const MAX_SPEECH_CHARS: usize = 4_096;
/// publik's published prices on 2026-10-19 (the `whisper-1` and `tts-1` lines of the price sheet),
/// for the estimate shown before a button. The charge line after the request is what was charged.
const TRANSCRIBE_MICROS_PER_MINUTE: i64 = 6_000;
const TRANSCRIBE_RATE: &str = "$0.006 a minute";
const SPEECH_MICROS_PER_1K_CHARS: i64 = 15_000;
const SPEECH_RATE: &str = "$0.015 for 1,000 characters";
/// The speech line's voices; the first is the crate's default.
pub const VOICES: [(&str, &str); 6] =
    [("alloy", "Alloy"), ("echo", "Echo"), ("fable", "Fable"), ("onyx", "Onyx"), ("nova", "Nova"), ("shimmer", "Shimmer")];
/// File names of speech clips: `AI Speech <n>.mp3`.
const SPEECH_BASE: &str = "AI Speech";
/// The undo step all markers of one transcription share.
const MARKERS_STEP: &str = "ai.transcribe.markers";

const OFF: &str = "AI is off in this session.";
const TRANSCRIBING: &str = "Transcribing… This can take a minute.";
const GENERATING: &str = "Generating speech… This can take a minute.";
const STOPPED: &str = "The request stopped unexpectedly. Try again.";

/// AI state of the desktop app (`SoundApp::ai`).
pub struct Ai {
    pub state: AiState,
    frames: u32,
    first_run_checked: bool,
    /// The setup window is wanted; it opens once nothing else asks for the person's attention.
    open_setup: bool,
    setup_window: bool,
    pub transcribe_window: bool,
    pub speech_window: bool,
    pub prefs_window: bool,
    /// The Preferences window shows the setup flow in place of the AI status.
    inline_setup: bool,
    /// Transcribe the selected clip (else the session mix of the selection) when both are there.
    use_clip: bool,
    /// Bumps each time the open session is replaced (new, open, close, revert, recover), so markers
    /// never land in a session that the audio did not come from.
    serial: u64,
    /// The transcription audio being rendered, a slice per frame.
    prep: Option<Prep>,
    transcribe: Task<Receiver<Result<(Transcript, Reply), Fail>>, Transcript>,
    speech: Task<Receiver<Result<(SpeechOut, Reply), Fail>>, SpeechOut>,
    /// The speech text and voice (kept between openings of the window).
    pub text: String,
    pub voice_id: String,
}

/// One feature's request in flight, its result waiting to be placed, and what the window says.
struct Task<J, O> {
    job: Option<J>,
    waiting: Option<O>,
    note: Option<String>,
    error: Option<String>,
}

impl<J, O> Default for Task<J, O> {
    fn default() -> Self {
        Task { job: None, waiting: None, note: None, error: None }
    }
}

impl<J, O> Task<J, O> {
    fn busy(&self) -> bool {
        self.job.is_some()
    }
    fn start(&mut self, job: J) {
        self.job = Some(job);
        self.note = None;
        self.error = None;
    }
}

/// The audio a transcription is made of: one clip alone, or the session mix, over `range`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Source {
    /// The audio clip to render alone (its track and id); `None` = the session mix.
    clip: Option<(TrackId, ClipId)>,
    range: Range,
    /// What the window says about it.
    label: String,
}

/// A part of the transcription: seconds from the start of the audio, and its words.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Segment {
    at: f64,
    text: String,
}

/// A finished transcription: where it goes and what came back.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Transcript {
    /// [`Ai::serial`] when the audio was taken.
    serial: u64,
    /// Session time of the start of the audio, in seconds.
    start: f64,
    segments: Vec<Segment>,
    /// The whole text, for Copy Text.
    text: String,
    /// How far into the audio the markers reach (the end of the last part that came back), and
    /// the audio's length, in seconds.
    done: f64,
    length: f64,
    /// Why the rest did not come back, when a later part failed after earlier parts were charged.
    stopped: Option<Fail>,
}

/// A finished speech clip: the audio file's bytes, what they decode to, and where they were saved
/// once they were.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpeechOut {
    bytes: Vec<u8>,
    ext: &'static str,
    /// Frames, sample rate and channels of the audio, when SoundCraft could read it.
    shape: Option<(u64, u32, usize)>,
    path: Option<String>,
}

impl Ai {
    pub fn new(app: AppInfo) -> Self {
        Ai {
            state: AiState::new(app),
            frames: 0,
            first_run_checked: false,
            open_setup: false,
            setup_window: false,
            transcribe_window: false,
            speech_window: false,
            prefs_window: false,
            inline_setup: false,
            use_clip: true,
            serial: 0,
            prep: None,
            transcribe: Task::default(),
            speech: Task::default(),
            text: String::new(),
            voice_id: VOICES[0].0.to_string(),
        }
    }

    /// Do not open the setup window on this launch (an agent drives the app over `--control`).
    /// Preferences and the AI windows still offer it.
    pub fn skip_first_run(&mut self) {
        self.first_run_checked = true;
    }

    /// A transcription (its audio or its request) or a speech clip is on its way.
    pub fn working(&self) -> bool {
        self.prep.is_some() || self.transcribe.busy() || self.speech.busy()
    }
}

/// `PUBLIK_SELF_TEST=1`: provision for real and print one line (publikhq.com/skills/setup.md).
pub fn self_test(app: &AppInfo) -> String {
    craft_publik::self_test(app)
}

fn local(message: impl Into<String>) -> Fail {
    Fail { kind: "local".into(), message: message.into(), ..Fail::default() }
}

/// Run a background request; a panic in it becomes a failure the window shows.
fn guarded<T>(work: impl FnOnce() -> Result<T, Fail>) -> Result<T, Fail> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|_| Err(local(STOPPED)))
}

// ---- once a frame -------------------------------------------------------------------------------

/// Once a frame, from `SoundApp::logic` before the transport: note a replaced session, render a
/// slice of the transcription audio, take finished requests (and place what they made), and open
/// the setup window on the first launch once nothing else asks for the person's attention.
pub fn frame(app: &mut SoundApp, ctx: &egui::Context) {
    let replaced = app.engine.transport_requests.iter().any(|r| matches!(r, TransportRequest::ResetSession));
    let quiet = app.dialogs.open.is_none() && app.leftovers.is_empty() && app.bounce.is_none();
    let Some(ai) = app.ai.as_mut() else { return };
    if replaced {
        ai.serial = ai.serial.wrapping_add(1);
        if ai.prep.take().is_some() {
            ai.transcribe.note = Some("The session closed before the audio was ready. Nothing was sent.".into());
        }
    }
    ai.state.poll();
    step_prep(ai, ctx);
    let transcript = take(&mut ai.transcribe, &mut ai.state);
    let speech = take(&mut ai.speech, &mut ai.state);
    ai.frames = ai.frames.saturating_add(1);
    if !ai.first_run_checked && ai.frames > 3 {
        ai.first_run_checked = true;
        ai.open_setup = ai.state.wants_first_run() || ai.state.owes_first_card();
    }
    if ai.open_setup && quiet {
        ai.open_setup = false;
        open_setup(ai);
    }
    if let Some(t) = transcript {
        arrived_transcript(app, t);
    }
    if let Some(v) = speech {
        arrived_speech(app, v);
    }
}

/// A finished request, if one came in: its charge noted, its failure shown.
fn take<O>(task: &mut Task<Receiver<Result<(O, Reply), Fail>>, O>, state: &mut AiState) -> Option<O> {
    let rx = task.job.as_ref()?;
    let r = match rx.try_recv() {
        Ok(r) => r,
        Err(TryRecvError::Empty) => return None,
        Err(TryRecvError::Disconnected) => Err(local(STOPPED)),
    };
    task.job = None;
    match r {
        Ok((out, reply)) => {
            state.note_reply(&reply);
            Some(out)
        }
        Err(f) => {
            // a 402 or a retired key is the result bar's (one message, one link); the rest is ours
            state.note_fail(&f);
            task.error = state.notice.is_none().then(|| f.message.clone());
            None
        }
    }
}

fn open_setup(ai: &mut Ai) {
    ai.state.begin();
    ai.setup_window = true;
}

/// Markers came in: they go in at once when they can; else they wait behind Place Markers.
fn arrived_transcript(app: &mut SoundApp, t: Transcript) {
    if let Some(f) = &t.stopped
        && let Some(ai) = app.ai.as_mut()
    {
        // the parts that came back are paid for and go in; the failure shows as usual
        ai.state.note_fail(f);
        ai.transcribe.error = ai.state.notice.is_none().then(|| f.message.clone());
    }
    let window = app.ai.as_ref().is_some_and(|ai| ai.transcribe_window);
    if t.segments.is_empty() {
        let note = if t.text.is_empty() {
            "The service found no speech in this audio. No markers were made."
        } else {
            "The service sent the text without times, so SoundCraft cannot make markers from it. Copy the text instead."
        };
        if !window {
            app.ui.status = format!("{note} (Clip › AI › Transcribe to Markers)");
        }
        if let Some(ai) = app.ai.as_mut() {
            ai.transcribe.note = Some(note.into());
            if !t.text.is_empty() {
                ai.transcribe.waiting = Some(t);
            }
        }
        return;
    }
    match place_markers(app, &t) {
        Ok(note) => {
            if !window {
                app.ui.status = note.clone();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.transcribe.note = Some(note);
            }
        }
        Err(why) => {
            if !window {
                app.ui.status = "The markers are ready. Open Clip › AI › Transcribe to Markers and click Place Markers.".into();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.transcribe.note = Some(waiting_note("The markers are ready, but they are not in the session yet.", &why, "Click Place Markers."));
                ai.transcribe.waiting = Some(t);
            }
        }
    }
}

fn arrived_speech(app: &mut SoundApp, mut v: SpeechOut) {
    let placed = place_speech(app, &mut v);
    let window = app.ai.as_ref().is_some_and(|ai| ai.speech_window);
    match placed {
        Ok(note) => {
            if !window {
                app.ui.status = note.clone();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.speech.note = Some(note);
            }
        }
        Err(why) => {
            if !window {
                app.ui.status = "The speech clip is ready. Open Clip › AI › Text to Speech Clip and click Place Clip.".into();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.speech.note = Some(waiting_note("The speech clip is ready, but it is not on the timeline yet.", &why, "Click Place Clip."));
                ai.speech.waiting = Some(v);
            }
        }
    }
}

/// "<ready> <why it is not in yet>. <what to do>"
fn waiting_note(ready: &str, why: &str, todo: &str) -> String {
    let mut s = ready.to_string();
    let why = why.trim().trim_end_matches('.');
    if !why.is_empty() {
        s.push(' ');
        s.push_str(why);
        s.push('.');
    }
    s.push(' ');
    s.push_str(todo);
    s
}

// ---- the menu -----------------------------------------------------------------------------------

/// Whether an AI menu item is enabled (`None` for other ids): only where AI is on.
pub fn item_enabled(app: &SoundApp, id: &str) -> Option<bool> {
    matches!(id, TRANSCRIBE | SPEECH | PREFERENCES).then_some(app.ai.is_some())
}

/// Run an AI window command (`None` for other ids): show or hide its window, like the app's other
/// `window.*` commands (`{"value": bool}`, else a toggle). Nothing is sent here.
pub fn run(app: &mut SoundApp, id: &str, p: &Value) -> Option<Result<Value, String>> {
    if !matches!(id, TRANSCRIBE | SPEECH | PREFERENCES) {
        return None;
    }
    let Some(ai) = app.ai.as_mut() else { return Some(Err(OFF.into())) };
    let open = match id {
        TRANSCRIBE => &mut ai.transcribe_window,
        SPEECH => &mut ai.speech_window,
        _ => &mut ai.prefs_window,
    };
    *open = p.get("value").and_then(Value::as_bool).unwrap_or(!*open);
    let shown = *open;
    if id == TRANSCRIBE && shown {
        // the clip when one is selected, else the selection
        ai.use_clip = true;
    }
    Some(Ok(json!({"value": shown})))
}

// ---- what a transcription sends -----------------------------------------------------------------

/// The selected clip: the first audio clip the Grabber selected.
pub(crate) fn clip_source(s: &Session) -> Option<Source> {
    let (track, clip) = s.edit.selected_clips.iter().find_map(|id| {
        let (t, c) = s.find_clip(*id)?;
        c.is_audio().then_some((t, c))
    })?;
    let track_name = s.track(track).map_or("", |t| t.name.as_str());
    let name = if clip.name.trim().is_empty() { "Clip" } else { clip.name.as_str() };
    Some(Source { clip: Some((track, clip.id)), range: clip.range(), label: format!("The selected clip: {name} on {track_name}.") })
}

/// The session mix from the selection's start to its end, when the selection is a range.
pub(crate) fn range_source(s: &Session) -> Option<Source> {
    let r = s.edit.selection;
    (!r.is_empty() && r.start >= 0).then(|| Source {
        clip: None,
        range: r,
        label: format!("The session mix from {} to {}.", clock(s.sample_rate.seconds(r.start)), clock(s.sample_rate.seconds(r.end))),
    })
}

/// `m:ss` (or `h:mm:ss`) for a length in seconds.
pub(crate) fn clock(seconds: f64) -> String {
    let s = if seconds.is_finite() { seconds.max(0.0).round() as i64 } else { 0 };
    let (h, m, sec) = (s / 3600, (s / 60) % 60, s % 60);
    if h > 0 { format!("{h}:{m:02}:{sec:02}") } else { format!("{m}:{sec:02}") }
}

/// How many parts (requests) `seconds` of audio go out in.
fn parts_for(seconds: f64) -> usize {
    let part = PART_SECONDS as f64;
    let tail = TAIL_SECONDS as f64;
    if !seconds.is_finite() || seconds <= part + tail { 1 } else { ((seconds - tail) / part).ceil().max(1.0) as usize }
}

/// The estimate before Transcribe.
pub(crate) fn transcribe_estimate(seconds: f64) -> String {
    let micros = (seconds.max(0.0) * TRANSCRIBE_MICROS_PER_MINUTE as f64 / 60.0).ceil() as i64;
    format!(
        "publik API charges for each minute of audio, about {TRANSCRIBE_RATE}. This audio costs about {}. The exact charge shows after the request.",
        craft_publik::dollars(micros)
    )
}

/// The estimate before Generate.
pub(crate) fn speech_estimate(chars: usize) -> String {
    let chars = i64::try_from(chars).unwrap_or(i64::MAX);
    let micros = chars.saturating_mul(SPEECH_MICROS_PER_1K_CHARS).saturating_add(999) / 1000;
    format!(
        "publik API charges for each character, about {SPEECH_RATE}. This text costs about {}. The exact charge shows after the request.",
        craft_publik::dollars(micros)
    )
}

/// The cost lines before a button that sends: publik's estimate and why it costs, or who bills an
/// own key.
fn cost_lines(state: &AiState, provider: &Provider, estimate: String) -> Vec<String> {
    match provider {
        Provider::Publik { .. } => {
            let why = match &state.credential {
                Some(c) if !c.cost_sentence.trim().is_empty() => c.cost_sentence.clone(),
                _ => WHY_IT_COSTS.to_string(),
            };
            vec![estimate, why]
        }
        Provider::Own { service, .. } => vec![format!("Your own key's service ({}) bills this request, not publik.", service.name())],
        Provider::NotSetUp => Vec::new(),
    }
}

fn too_long(seconds: f64) -> String {
    format!("This audio is {} long. One click can send {} at most. Select a shorter clip or range.", clock(seconds), clock(MAX_TRANSCRIBE_SECONDS))
}

/// What renders the audio: the session mix (as Bounce Mix renders it) or one clip alone (as Export
/// Clips as Files renders it).
enum Render {
    Mix(Box<soundcraft_mix::RangeRender>),
    Clip { track: TrackId, start: Samples },
}

/// One part of the audio a transcription sends: where it starts in the audio, and its samples at
/// [`SEND_RATE`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Part {
    offset: f64,
    seconds: f64,
    samples: Vec<f32>,
}

/// The transcription audio, rendered a slice per frame on the UI thread (as Bounce Mix renders),
/// mixed to mono and cut into parts at quiet moments.
pub(crate) struct Prep {
    session: Arc<Session>,
    render: Render,
    rate: u32,
    /// Samples rendered so far and in all, at the session rate.
    done: usize,
    total: usize,
    /// Rendered samples not yet in a part, and where they start in the audio.
    pending: Vec<f32>,
    pending_at: usize,
    parts: Vec<Part>,
    /// Part, search and tail lengths, in samples at the session rate.
    part_len: usize,
    search_len: usize,
    tail_len: usize,
    /// What the request needs once the audio is ready.
    provider: Provider,
    serial: u64,
    start: f64,
}

enum PrepStep {
    More,
    Done,
}

impl Prep {
    pub(crate) fn new(session: &Arc<Session>, src: &Source, provider: Provider, serial: u64) -> Result<Prep, String> {
        let rate = session.sample_rate.hz().max(1);
        let total = usize::try_from(src.range.len()).unwrap_or(0);
        let (session, render) = match src.clip {
            Some((track, id)) => {
                // the clip alone on its track, as Export Clips as Files renders it
                let mut solo = (**session).clone();
                let clip = solo.find_clip(id).map(|(_, c)| c.clone()).ok_or("The clip was deleted.")?;
                let playlist = solo.track_mut(track).and_then(|t| t.playlist_mut()).ok_or("The clip's track was deleted.")?;
                playlist.clips = vec![clip];
                (Arc::new(solo), Render::Clip { track, start: src.range.start })
            }
            None => (Arc::clone(session), Render::Mix(Box::new(soundcraft_mix::RangeRender::new(session, None, src.range, 1024)))),
        };
        let at = |secs: usize| secs.saturating_mul(rate as usize);
        Ok(Prep {
            session,
            render,
            rate,
            done: 0,
            total,
            pending: Vec::new(),
            pending_at: 0,
            parts: Vec::new(),
            part_len: at(PART_SECONDS),
            search_len: at(SEARCH_SECONDS),
            tail_len: at(TAIL_SECONDS),
            provider,
            serial,
            start: src.range.start as f64 / f64::from(rate),
        })
    }

    /// How far the rendering got, from 0 to 1.
    fn progress(&self) -> f32 {
        (self.done as f64 / self.total.max(1) as f64).min(1.0) as f32
    }

    /// Renders blocks until `more` returns false (at least one per call).
    fn step(&mut self, mut more: impl FnMut() -> bool) -> PrepStep {
        loop {
            let block = match &mut self.render {
                Render::Mix(r) => r.next_block(&self.session).map(|(ch, n)| mono(ch, n)),
                Render::Clip { track, start } => (self.done < self.total).then(|| {
                    let n = (self.total - self.done).min(self.rate as usize);
                    let at = start.saturating_add(i64::try_from(self.done).unwrap_or(i64::MAX));
                    let audio = soundcraft_mix::render_clips(&self.session, *track, Range::new(at, at.saturating_add(n as i64)));
                    mono(&audio, n)
                }),
            };
            let Some(block) = block else {
                self.finish();
                return PrepStep::Done;
            };
            self.done = self.done.saturating_add(block.len());
            self.pending.extend_from_slice(&block);
            self.cut_parts();
            if !more() {
                return PrepStep::More;
            }
        }
    }

    /// Cut full parts off the rendered audio, each at the quietest moment of its last seconds.
    fn cut_parts(&mut self) {
        while self.pending.len() >= self.part_len.max(1) {
            if self.total.saturating_sub(self.pending_at) <= self.part_len.saturating_add(self.tail_len) {
                // the rest is one last part
                return;
            }
            let from = self.part_len.saturating_sub(self.search_len);
            let cut = quietest(&self.pending, from, self.part_len, (self.rate / 50) as usize);
            let head: Vec<f32> = self.pending.drain(..cut).collect();
            self.push_part(head);
        }
    }

    fn finish(&mut self) {
        let rest = std::mem::take(&mut self.pending);
        // shorter than the transcription line takes: nothing a word fits in
        if rest.len() as f64 >= 0.1 * f64::from(self.rate) || self.parts.is_empty() {
            self.push_part(rest);
        }
    }

    fn push_part(&mut self, samples: Vec<f32>) {
        let n = samples.len();
        if n == 0 {
            return;
        }
        let rate = f64::from(self.rate);
        let at16 = soundcraft_dsp::offline::resample(&[samples], self.rate, SEND_RATE).into_iter().next().unwrap_or_default();
        self.parts.push(Part { offset: self.pending_at as f64 / rate, seconds: n as f64 / rate, samples: at16 });
        self.pending_at = self.pending_at.saturating_add(n);
    }
}

/// The mono average of the first `n` samples of planar channels (missing samples read as silence).
fn mono(channels: &[Vec<f32>], n: usize) -> Vec<f32> {
    let k = 1.0 / channels.len().max(1) as f32;
    (0..n).map(|i| channels.iter().map(|c| c.get(i).copied().unwrap_or(0.0)).sum::<f32>() * k).collect()
}

/// Where to cut `x` between `from` and `to`: the middle of its quietest `win` samples there.
fn quietest(x: &[f32], from: usize, to: usize, win: usize) -> usize {
    let to = to.min(x.len());
    let win = win.max(2);
    let mut best = (f32::INFINITY, to);
    let mut i = from.min(to);
    while i.saturating_add(win) <= to {
        let energy: f32 = x.get(i..i + win).map_or(f32::INFINITY, |w| w.iter().map(|v| v * v).sum());
        if energy < best.0 {
            best = (energy, i + win / 2);
        }
        i += win / 2;
    }
    best.1.clamp(1, x.len().max(1))
}

/// A part as 16-bit FLAC at [`SEND_RATE`] (SoundCraft's own encoder).
pub(crate) fn flac(samples: &[f32]) -> Result<Vec<u8>, String> {
    let buf = soundcraft_audio_io::AudioBuffer { sample_rate: SEND_RATE, channels: vec![samples.to_vec()] };
    let opts = soundcraft_audio_io::EncodeOptions {
        format: soundcraft_audio_io::FileFormat::Flac,
        bit_depth: soundcraft_audio_io::BitDepth::Int16,
        dither: false,
        bwf: None,
    };
    soundcraft_audio_io::encode(&buf, &opts).map_err(|e| e.to_string())
}

/// The segments of one part (`client::transcribe` hands them over as JSON in `Reply::bytes`),
/// clipped to the part's `length` and moved by its `offset`; segments without time or words are
/// dropped.
pub(crate) fn parse_segments(bytes: &[u8], offset: f64, length: f64) -> Vec<Segment> {
    let max = if length.is_finite() { length.max(0.0) } else { 0.0 };
    let v: Value = serde_json::from_slice(bytes).unwrap_or(Value::Null);
    let Some(list) = v.as_array() else { return Vec::new() };
    list.iter()
        .filter_map(|s| {
            let start = s.get("start")?.as_f64()?;
            let text = clean_text(s.get("text")?.as_str()?);
            (start.is_finite() && !text.is_empty() && start < max).then(|| Segment { at: offset + start.max(0.0), text })
        })
        .collect()
}

/// One line of text: runs of white space (line breaks too) become one space.
fn clean_text(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A marker name: the segment's words, cut to [`MAX_NAME_CHARS`].
fn marker_name(text: &str) -> String {
    if text.chars().count() <= MAX_NAME_CHARS {
        return text.to_string();
    }
    let mut s: String = text.chars().take(MAX_NAME_CHARS - 1).collect();
    s.push('…');
    s
}

/// Temporary files of one request (the audio parts sent), removed with it.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("soundcraft-ai-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Scratch(dir))
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Add one part's answer to the answer of the request: the charges add up, the balance is the last.
fn add_reply(total: &mut Reply, r: &Reply) {
    total.charge_micros = match (total.charge_micros, r.charge_micros) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (a, b) => a.or(b),
    };
    if r.balance_micros.is_some() {
        total.balance_micros = r.balance_micros;
    }
    if r.claim_state.is_some() {
        total.claim_state.clone_from(&r.claim_state);
    }
    if r.model_served.is_some() {
        total.model_served.clone_from(&r.model_served);
    }
}

/// Pack and send the parts one after the other; read the segments that come back. Runs on a
/// thread. A part that fails after earlier parts came back ends the request with what came back.
fn transcribe_work(provider: &Provider, parts: &[Part], serial: u64, start: f64, length: f64) -> Result<(Transcript, Reply), Fail> {
    let scratch = Scratch::new().map_err(local)?;
    let mut out = Transcript { serial, start, segments: Vec::new(), text: String::new(), done: 0.0, length, stopped: None };
    let mut total = Reply::default();
    for (k, part) in parts.iter().enumerate() {
        let sent = flac(&part.samples)
            .map_err(|e| local(format!("SoundCraft could not prepare the audio: {e}")))
            .and_then(|bytes| scratch.write(&format!("part-{k}.flac"), &bytes).map_err(local))
            .and_then(|path| client::transcribe(provider, &path, ""));
        let reply = match sent {
            Ok(reply) => reply,
            Err(f) if k == 0 => return Err(f),
            Err(f) => {
                out.stopped = Some(f);
                break;
            }
        };
        add_reply(&mut total, &reply);
        let room = MAX_MARKERS.saturating_sub(out.segments.len());
        out.segments.extend(parse_segments(&reply.bytes, part.offset, part.seconds).into_iter().take(room));
        let text = clean_text(&reply.text);
        if !text.is_empty() {
            if !out.text.is_empty() {
                out.text.push(' ');
            }
            out.text.push_str(&text);
        }
        out.done = part.offset + part.seconds;
    }
    Ok((out, total))
}

/// The Transcribe button: check what is sent, then render its audio a slice per frame.
fn start_transcribe(app: &mut SoundApp, src: Source) -> Result<(), String> {
    let seconds = app.engine.session().sample_rate.seconds(src.range.len());
    if seconds < MIN_TRANSCRIBE_SECONDS {
        return Err("The audio is too short. Select at least half a second.".into());
    }
    if seconds > MAX_TRANSCRIBE_SECONDS {
        return Err(too_long(seconds));
    }
    let ai = app.ai.as_ref().ok_or(OFF)?;
    let provider = ai.state.provider();
    if !provider.supports(Line::Transcribe) {
        return Err(Fail::unsupported(Line::Transcribe).message);
    }
    let serial = ai.serial;
    if src.clip.is_none() {
        // Hosted plugins keep state the session cannot see: store it first, as Bounce Mix does.
        app.capture_plugin_states();
    }
    let session = app.engine.session_arc();
    let prep = Prep::new(&session, &src, provider, serial)?;
    let ai = app.ai.as_mut().ok_or(OFF)?;
    ai.transcribe.note = None;
    ai.transcribe.error = None;
    ai.prep = Some(prep);
    Ok(())
}

/// Render the transcription audio for a moment; once it is ready, send it on a thread.
fn step_prep(ai: &mut Ai, ctx: &egui::Context) {
    let Some(prep) = &mut ai.prep else { return };
    let until = std::time::Instant::now() + std::time::Duration::from_millis(30);
    if let PrepStep::More = prep.step(|| std::time::Instant::now() < until) {
        ctx.request_repaint();
        return;
    }
    let Some(prep) = ai.prep.take() else { return };
    let length = prep.total as f64 / f64::from(prep.rate);
    let Prep { parts, provider, serial, start, .. } = prep;
    // Silence holds no words: it is not sent, so it costs nothing.
    let parts: Vec<Part> = parts.into_iter().filter(|p| !silent(&p.samples)).collect();
    if parts.is_empty() {
        ai.transcribe.error = Some("This audio is silent. Nothing was sent.".into());
        return;
    }
    let rx = spawn(ctx, move || guarded(|| transcribe_work(&provider, &parts, serial, start, length)));
    ai.transcribe.start(rx);
}

/// No sample of the part reaches -60 dBFS.
fn silent(samples: &[f32]) -> bool {
    samples.iter().all(|v| v.abs() < 0.001)
}

/// The markers at their segments' starts, through `markers.add`, all in one undo step. Returns
/// what the window says.
pub(crate) fn place_markers(app: &mut SoundApp, t: &Transcript) -> Result<String, String> {
    if app.ai.as_ref().is_some_and(|ai| ai.serial != t.serial) {
        return Err("The session that the audio came from is closed".into());
    }
    if app.engine.transport.recording {
        return Err("Stop recording first".into());
    }
    if app.is_playing() {
        return Err("Stop playback first".into());
    }
    if t.segments.is_empty() {
        return Err("There are no markers to place".into());
    }
    let rate = app.engine.session().sample_rate;
    app.engine.end_merge();
    let mut placed = 0usize;
    for seg in &t.segments {
        let at = rate.samples(t.start + seg.at).max(0);
        let params = json!({"at": at, "name": marker_name(&seg.text)});
        if let Err(e) = app.engine.execute_merged("markers.add", &params, MARKERS_STEP) {
            app.engine.end_merge();
            if placed > 0 {
                // all or nothing: the step holds the markers placed so far
                app.engine.undo();
            }
            return Err(e.to_string());
        }
        placed += 1;
    }
    app.engine.end_merge();
    let mut note = if placed == 1 {
        "1 marker is on the Markers ruler. Edit › Undo removes it.".to_string()
    } else {
        format!("{placed} markers are on the Markers ruler. Edit › Undo removes them.")
    };
    if t.stopped.is_some() && t.done < t.length {
        note.push_str(&format!(" They cover the first {} of {}: the request stopped there.", clock(t.done), clock(t.length)));
    }
    Ok(note)
}

fn place_waiting_markers(app: &mut SoundApp) {
    let Some(t) = app.ai.as_mut().and_then(|ai| ai.transcribe.waiting.take()) else { return };
    let placed = place_markers(app, &t);
    let Some(ai) = app.ai.as_mut() else { return };
    match placed {
        Ok(note) => ai.transcribe.note = Some(note),
        Err(why) => {
            ai.transcribe.note = Some(waiting_note("The markers are ready, but they are not in the session yet.", &why, "Click Place Markers."));
            ai.transcribe.waiting = Some(t);
        }
    }
}

// ---- text to speech -----------------------------------------------------------------------------

/// The file type of audio bytes, by their first bytes (the speech line answers MP3).
fn audio_ext(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"RIFF") {
        "wav"
    } else if bytes.starts_with(b"OggS") {
        "ogg"
    } else if bytes.starts_with(b"fLaC") {
        "flac"
    } else {
        "mp3"
    }
}

fn start_speech(app: &mut SoundApp, ctx: &egui::Context) -> Result<(), String> {
    let ai = app.ai.as_mut().ok_or(OFF)?;
    let text = ai.text.trim().to_string();
    if text.is_empty() {
        return Err("Type the text first.".into());
    }
    if text.chars().count() > MAX_SPEECH_CHARS {
        return Err(format!("The text is too long. One request takes {MAX_SPEECH_CHARS} characters at most."));
    }
    let provider = ai.state.provider();
    if !provider.supports(Line::Speech) {
        return Err(Fail::unsupported(Line::Speech).message);
    }
    let voice = VOICES.iter().find(|(id, _)| *id == ai.voice_id).map_or(VOICES[0].0, |(id, _)| *id);
    let rx = spawn(ctx, move || {
        guarded(|| {
            let mut reply = client::speech(&provider, &text, voice)?;
            let bytes = std::mem::take(&mut reply.bytes);
            if bytes.is_empty() {
                return Err(local("The service sent no audio. Try again."));
            }
            let ext = audio_ext(&bytes);
            let shape = soundcraft_audio_io::decode(&bytes, Some(ext)).ok().map(|(_, b)| (b.frames() as u64, b.sample_rate, b.num_channels()));
            Ok((SpeechOut { bytes, ext, shape, path: None }, reply))
        })
    });
    ai.speech.start(rx);
    Ok(())
}

/// Where speech clips are saved: where recordings go (`Audio Files` next to the saved session,
/// else in the session's autosave folder), else a SoundCraft folder in the temporary folder.
fn speech_dir(app: &SoundApp) -> PathBuf {
    if let Some(dir) = app.engine.path.as_deref().and_then(|p| Path::new(p).parent()).filter(|d| !d.as_os_str().is_empty()) {
        return dir.join("Audio Files");
    }
    let name = soundcraft_engine::io::sanitize_name(&app.engine.session().name);
    match &app.autosave_dir {
        Some(root) => root.join(name).join("Audio Files"),
        None => std::env::temp_dir().join("SoundCraft").join(name).join("Audio Files"),
    }
}

/// `<dir>/AI Speech <n>.<ext>` with the first `n` used by neither a file nor an audio file of the
/// session.
fn speech_path(s: &Session, dir: &Path, ext: &str) -> Result<PathBuf, String> {
    (1..=9_999u32)
        .map(|k| format!("{SPEECH_BASE} {k}"))
        .find(|stem| !s.sources.iter().any(|x| &x.name == stem) && !dir.join(format!("{stem}.{ext}")).exists())
        .map(|stem| dir.join(format!("{stem}.{ext}")))
        .ok_or_else(|| format!("{}: no free file name for \"{SPEECH_BASE} <n>.{ext}\"", dir.display()))
}

/// The first audio track of `channels` channels with no clip in `at..at + len`: the selected tracks
/// first, then the tracks in order.
pub(crate) fn free_track(s: &Session, at: Samples, len: Samples, channels: usize) -> Option<TrackId> {
    let r = Range::new(at, at.saturating_add(len.max(1)));
    let fits = |id: TrackId| {
        s.track(id).is_some_and(|t| {
            t.kind == TrackKind::Audio && !t.inactive && !t.hidden && t.channels() == channels && !t.clips().iter().any(|c| c.range().overlaps(&r))
        })
    };
    s.edit.selected_tracks.iter().copied().find(|id| fits(*id)).or_else(|| s.tracks.iter().map(|t| t.id).find(|id| fits(*id)))
}

/// The speech saved and on the timeline at the edit cursor through `file.import_audio` (one undo
/// step). The saved path is kept, so a retry places the same file.
pub(crate) fn place_speech(app: &mut SoundApp, v: &mut SpeechOut) -> Result<String, String> {
    if app.engine.transport.recording {
        return Err("Stop recording first".into());
    }
    if app.is_playing() {
        return Err("Stop playback first".into());
    }
    let path = match v.path.clone().filter(|p| Path::new(p).is_file()) {
        Some(p) => p,
        None => {
            let dir = speech_dir(app);
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let path = speech_path(app.engine.session(), &dir, v.ext)?;
            std::fs::write(&path, &v.bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            let path = path.to_string_lossy().into_owned();
            v.path = Some(path.clone());
            path
        }
    };
    let s = app.engine.session();
    let at = s.edit.selection.start.max(0);
    let track = v.shape.and_then(|(frames, rate, channels)| {
        let len = (frames as f64 * s.sample_rate.as_f64() / f64::from(rate.max(1))).ceil() as Samples;
        free_track(s, at, len, channels)
    });
    let track_name = track.and_then(|t| s.track(t)).map(|t| t.name.clone());
    let mut params = json!({"path": path, "at": at});
    match track {
        Some(t) => params["track"] = json!(t.0),
        None => params["new_tracks"] = json!(true),
    }
    app.run("file.import_audio", params)?;
    let file = Path::new(&path).file_name().map_or_else(|| path.clone(), |f| f.to_string_lossy().into_owned());
    Ok(match track_name {
        Some(name) => format!("The speech clip is on {name} at the edit cursor ({file}). Edit › Undo removes it."),
        None => format!("The speech clip is on a new track at the edit cursor ({file}). Edit › Undo removes it."),
    })
}

fn place_waiting_speech(app: &mut SoundApp) {
    let Some(mut v) = app.ai.as_mut().and_then(|ai| ai.speech.waiting.take()) else { return };
    let placed = place_speech(app, &mut v);
    let Some(ai) = app.ai.as_mut() else { return };
    match placed {
        Ok(note) => ai.speech.note = Some(note),
        Err(why) => {
            ai.speech.note = Some(waiting_note("The speech clip is ready, but it is not on the timeline yet.", &why, "Click Place Clip."));
            ai.speech.waiting = Some(v);
        }
    }
}

// ---- the windows --------------------------------------------------------------------------------

/// What a click in a window asks for, done after the window is drawn.
#[derive(Default)]
struct Clicks {
    send: bool,
    place: bool,
    discard: bool,
    cancel: bool,
    setup: bool,
    url: Option<String>,
}

/// The AI windows, drawn with the app's other windows.
pub fn show(app: &mut SoundApp, ctx: &egui::Context) {
    setup_window(app, ctx);
    transcribe_window(app, ctx);
    speech_window(app, ctx);
    prefs_window(app, ctx);
}

fn window<'a>(title: &str, id: &str, ctx: &egui::Context) -> egui::Window<'a> {
    egui::Window::new(RichText::new(title).font(bold(13.0)))
        .id(egui::Id::new(id))
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
}

fn open_url(ctx: &egui::Context, url: &str) {
    ctx.open_url(egui::OpenUrl::new_tab(url));
}

fn setup_window(app: &mut SoundApp, ctx: &egui::Context) {
    let Some(ai) = app.ai.as_mut() else { return };
    if !ai.setup_window {
        return;
    }
    let mut open = true;
    let mut action = Action::None;
    window("Set Up AI", "ai-setup", ctx).open(&mut open).show(ctx, |ui| action = ai.state.setup_ui(ui));
    match action {
        Action::Close => ai.setup_window = false,
        Action::OpenUrl(url) => open_url(ctx, &url),
        Action::None => {}
    }
    if !open {
        ai.setup_window = false;
    }
}

fn dim(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).small().color(Tokens::current().text_dim)).wrap());
}

fn warn(ui: &mut egui::Ui, text: &str) {
    let c = ui.visuals().warn_fg_color;
    ui.add(egui::Label::new(RichText::new(text).color(c)).wrap());
}

/// The button that sends (or places): the app's accent, as in its other dialogs.
fn primary(ui: &mut egui::Ui, label: &str, enabled: bool) -> bool {
    let b = egui::Button::new(RichText::new(label).strong()).fill(Tokens::current().accent);
    ui.add_enabled(enabled, b).clicked()
}

/// What a window shows when AI is not set up: why, and Set Up AI….
fn needs_setup(ui: &mut egui::Ui, what: &str, clicks: &mut Clicks) {
    ui.add(egui::Label::new(format!("{what} needs AI. Set it up first: publik API (no key to paste) or your own key.")).wrap());
    ui.add_space(8.0);
    if primary(ui, "Set Up AI…", true) {
        clicks.setup = true;
    }
}

/// Under the buttons: the request in flight, its note or error, the result bar (a 402's one link,
/// or the charge) and what is in use.
fn status_rows<J, O>(ui: &mut egui::Ui, state: &mut AiState, task: &Task<J, O>, busy_text: &str, clicks: &mut Clicks) {
    if task.busy() {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(busy_text).color(Tokens::current().text_dim));
        });
    }
    if let Some(note) = &task.note {
        ui.add_space(6.0);
        ui.add(egui::Label::new(note.as_str()).wrap());
    }
    if let Some(e) = &task.error {
        ui.add_space(6.0);
        let c = ui.visuals().error_fg_color;
        ui.add(egui::Label::new(RichText::new(e.as_str()).color(c)).wrap());
    }
    ui.add_space(6.0);
    if let Action::OpenUrl(url) = state.result_bar(ui) {
        clicks.url = Some(url);
    }
    dim(ui, &state.status_line());
}

fn transcribe_window(app: &mut SoundApp, ctx: &egui::Context) {
    if !app.ai.as_ref().is_some_and(|ai| ai.transcribe_window) {
        return;
    }
    let clip = clip_source(app.engine.session());
    let range = range_source(app.engine.session());
    let rate = app.engine.session().sample_rate;
    let Some(ai) = app.ai.as_mut() else { return };
    let provider = ai.state.provider();
    let mut clicks = Clicks::default();
    let mut open = true;
    let mut chosen: Option<Source> = None;
    window("Transcribe to Markers", "ai-transcribe", ctx).open(&mut open).show(ctx, |ui| {
        ui.set_max_width(460.0);
        if !provider.is_set_up() {
            needs_setup(ui, "Transcribe to Markers", &mut clicks);
            return;
        }
        let preparing = ai.prep.as_ref().map(Prep::progress);
        let (busy, waiting) = (preparing.is_some() || ai.transcribe.busy(), ai.transcribe.waiting.is_some());
        ui.add(
            egui::Label::new(
                "Transcribe to Markers sends audio to the AI service. The text comes back as markers: one marker at the start of each spoken part, with the words as its name.",
            )
            .wrap(),
        );
        dim(ui, "SoundCraft sends the audio as 16 kHz mono FLAC, in parts of up to one minute. Nothing is sent until you click Transcribe.");
        ui.add_space(8.0);
        if clip.is_some() && range.is_some() {
            ui.add_enabled_ui(!busy && !waiting, |ui| {
                ui.horizontal(|ui| {
                    if ui.radio(ai.use_clip, "The selected clip").clicked() {
                        ai.use_clip = true;
                    }
                    if ui.radio(!ai.use_clip, "The session mix of the selection").clicked() {
                        ai.use_clip = false;
                    }
                });
            });
        }
        chosen = if ai.use_clip { clip.clone().or_else(|| range.clone()) } else { range.clone().or_else(|| clip.clone()) };
        let mut can_send = !busy && !waiting && provider.supports(Line::Transcribe);
        match &chosen {
            Some(src) => {
                ui.add(egui::Label::new(src.label.as_str()).wrap());
                let seconds = rate.seconds(src.range.len());
                let parts = parts_for(seconds);
                ui.label(if parts > 1 { format!("Length: {}. It goes out in {parts} parts.", clock(seconds)) } else { format!("Length: {}.", clock(seconds)) });
                if seconds > MAX_TRANSCRIBE_SECONDS {
                    warn(ui, &too_long(seconds));
                    can_send = false;
                }
                if seconds < MIN_TRANSCRIBE_SECONDS {
                    can_send = false;
                }
                if !provider.supports(Line::Transcribe) {
                    warn(ui, &Fail::unsupported(Line::Transcribe).message);
                }
                ui.add_space(4.0);
                dim(ui, "Language: the service finds it from the audio.");
                for line in cost_lines(&ai.state, &provider, transcribe_estimate(seconds)) {
                    dim(ui, &line);
                }
            }
            None => {
                warn(ui, "Select a clip with the Grabber, or make an edit selection on the timeline.");
                can_send = false;
            }
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if let Some(progress) = preparing {
                ui.add(egui::ProgressBar::new(progress).desired_width(200.0).text("Preparing the audio…"));
                if ui.button("Cancel").clicked() {
                    clicks.cancel = true;
                }
            } else if let Some(w) = &ai.transcribe.waiting {
                if primary(ui, "Place Markers", !w.segments.is_empty()) {
                    clicks.place = true;
                }
                if !w.text.is_empty() && ui.button("Copy Text").clicked() {
                    ui.ctx().copy_text(w.text.clone());
                }
                if ui.button("Discard").clicked() {
                    clicks.discard = true;
                }
            } else if primary(ui, "Transcribe", can_send) {
                clicks.send = true;
            }
        });
        status_rows(ui, &mut ai.state, &ai.transcribe, TRANSCRIBING, &mut clicks);
    });
    let Some(ai) = app.ai.as_mut() else { return };
    if !open {
        ai.transcribe_window = false;
    }
    if clicks.setup {
        open_setup(ai);
    }
    if clicks.cancel {
        ai.prep = None;
        ai.transcribe.note = Some("Cancelled. Nothing was sent.".into());
    }
    if clicks.discard {
        ai.transcribe.waiting = None;
        ai.transcribe.note = None;
    }
    if let Some(url) = clicks.url {
        open_url(ctx, &url);
    }
    if clicks.place {
        place_waiting_markers(app);
    }
    if clicks.send
        && let Some(src) = chosen
        && let Err(e) = start_transcribe(app, src)
        && let Some(ai) = app.ai.as_mut()
    {
        ai.transcribe.error = Some(e);
    }
}

fn speech_window(app: &mut SoundApp, ctx: &egui::Context) {
    if !app.ai.as_ref().is_some_and(|ai| ai.speech_window) {
        return;
    }
    let Some(ai) = app.ai.as_mut() else { return };
    let provider = ai.state.provider();
    let mut clicks = Clicks::default();
    let mut open = true;
    window("Text to Speech Clip", "ai-speech", ctx).open(&mut open).show(ctx, |ui| {
        ui.set_max_width(460.0);
        if !provider.is_set_up() {
            needs_setup(ui, "Text to Speech Clip", &mut clicks);
            return;
        }
        let (busy, waiting) = (ai.speech.busy(), ai.speech.waiting.is_some());
        ui.add(
            egui::Label::new(
                "Text to Speech Clip sends your text to the AI service. The speech comes back as a new audio clip at the edit cursor, on a free audio track.",
            )
            .wrap(),
        );
        dim(ui, "Nothing is sent until you click Generate.");
        ui.add_space(8.0);
        ui.label("Text");
        let edit = egui::TextEdit::multiline(&mut ai.text)
            .id(egui::Id::new("ai-speech-text"))
            .hint_text("Welcome back. In this part, we look at the first results.")
            .desired_rows(5)
            .desired_width(f32::INFINITY)
            .char_limit(MAX_SPEECH_CHARS);
        ui.add_enabled(!busy && !waiting, edit);
        let chars = ai.text.trim().chars().count();
        dim(ui, &format!("{chars} of {MAX_SPEECH_CHARS} characters."));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Voice");
            let shown = VOICES.iter().find(|(id, _)| *id == ai.voice_id).map_or(VOICES[0].1, |(_, name)| *name);
            ui.add_enabled_ui(!busy && !waiting, |ui| {
                egui::ComboBox::from_id_salt("ai-speech-voice").selected_text(shown).width(160.0).show_ui(ui, |ui| {
                    for (id, name) in VOICES {
                        if ui.selectable_label(ai.voice_id == id, name).clicked() {
                            ai.voice_id = id.to_string();
                        }
                    }
                });
            });
        });
        ui.add_space(4.0);
        if !provider.supports(Line::Speech) {
            warn(ui, &Fail::unsupported(Line::Speech).message);
        }
        for line in cost_lines(&ai.state, &provider, speech_estimate(chars)) {
            dim(ui, &line);
        }
        ui.add_space(8.0);
        let can_send = !busy && !waiting && chars > 0 && provider.supports(Line::Speech);
        ui.horizontal(|ui| {
            if waiting {
                if primary(ui, "Place Clip", true) {
                    clicks.place = true;
                }
                if ui.button("Discard").clicked() {
                    clicks.discard = true;
                }
            } else if primary(ui, "Generate", can_send) {
                clicks.send = true;
            }
        });
        status_rows(ui, &mut ai.state, &ai.speech, GENERATING, &mut clicks);
    });
    let Some(ai) = app.ai.as_mut() else { return };
    if !open {
        ai.speech_window = false;
    }
    if clicks.setup {
        open_setup(ai);
    }
    if clicks.discard {
        ai.speech.waiting = None;
        ai.speech.note = None;
    }
    if let Some(url) = clicks.url {
        open_url(ctx, &url);
    }
    if clicks.place {
        place_waiting_speech(app);
    }
    if clicks.send
        && let Err(e) = start_speech(app, ctx)
        && let Some(ai) = app.ai.as_mut()
    {
        ai.speech.error = Some(e);
    }
}

// ---- Setup › Preferences ------------------------------------------------------------------------

/// Setup › Preferences…: its AI section holds what is in use, the sign-in and key controls, and
/// the setup flow in place. The settings take effect at once.
fn prefs_window(app: &mut SoundApp, ctx: &egui::Context) {
    let Some(ai) = app.ai.as_mut() else { return };
    if !ai.prefs_window {
        return;
    }
    let mut open = true;
    let mut action = Action::None;
    window("Preferences", "ai-preferences", ctx).open(&mut open).show(ctx, |ui| {
        ui.set_max_width(460.0);
        ui.label(RichText::new("AI").font(bold(13.0)));
        ui.add_space(4.0);
        action = prefs_section(ai, ui);
    });
    match action {
        Action::OpenUrl(url) => open_url(ctx, &url),
        Action::Close | Action::None => {}
    }
    if !open {
        ai.prefs_window = false;
        ai.inline_setup = false;
    }
}

/// The AI section of Preferences.
fn prefs_section(ai: &mut Ai, ui: &mut egui::Ui) -> Action {
    dim(ui, "Clip › AI › Transcribe to Markers and Text to Speech Clip use publik API or your own key. Changes here take effect at once.");
    ui.add_space(6.0);
    if ai.inline_setup {
        let action = ai.state.setup_ui(ui);
        if action == Action::Close {
            ai.inline_setup = false;
        }
        return action;
    }
    let mut open_setup = false;
    let action = ai.state.options_ui(ui, &mut open_setup);
    if open_setup {
        ai.inline_setup = true;
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Services, SoundApp};

    fn demo_app() -> SoundApp {
        SoundApp::new(soundcraft_engine::demo::demo_engine(), None, Services::default())
    }

    fn with_ai(app: &mut SoundApp, name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("soundcraft-ai-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let info = AppInfo {
            slug: "soundcraft-test",
            name: "SoundCraft",
            version: "0.0.0",
            app_token: "",
            dialects: &["audio_transcriptions", "audio_speech"],
            config_dir: Some(dir.join("config")),
        };
        app.ai = Some(Ai::new(info));
        app.autosave_dir = Some(dir.join("Autosave"));
        dir
    }

    fn render_all(prep: &mut Prep) {
        for _ in 0..100_000 {
            if let PrepStep::Done = prep.step(|| true) {
                return;
            }
        }
        panic!("the audio never finished rendering");
    }

    fn tone_wav(seconds: f32, rate: u32) -> Vec<u8> {
        let n = (seconds * rate as f32) as usize;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let buf = soundcraft_audio_io::AudioBuffer { sample_rate: rate, channels: vec![samples] };
        soundcraft_audio_io::encode(&buf, &soundcraft_audio_io::EncodeOptions::default()).unwrap()
    }

    #[test]
    fn segments_are_read_clipped_moved_and_cleaned() {
        let bytes = br#"[
            {"id":0,"start":0.0,"end":2.5,"text":" Hello there."},
            {"id":1,"start":2.5,"end":4.0,"text":"  Two\n lines  "},
            {"id":2,"start":7.0,"end":9.0,"text":"past the end"},
            {"id":3,"start":1.0,"end":2.0,"text":"   "},
            {"id":4,"start":"x","end":2.0,"text":"bad time"},
            {"id":5,"start":-3.0,"end":0.5,"text":"before zero"}
        ]"#;
        let s = parse_segments(bytes, 60.0, 5.0);
        assert_eq!(s.len(), 3, "{s:?}");
        assert_eq!(s[0], Segment { at: 60.0, text: "Hello there.".into() });
        assert_eq!(s[1], Segment { at: 62.5, text: "Two lines".into() });
        assert_eq!(s[2].at, 60.0);
        assert!(parse_segments(b"null", 0.0, 5.0).is_empty());
        assert!(parse_segments(b"not json", 0.0, 5.0).is_empty());
        assert!(parse_segments(br#"[{"start":0,"end":1,"text":"x"}]"#, 0.0, f64::NAN).is_empty());
    }

    #[test]
    fn long_names_are_cut() {
        assert_eq!(marker_name("Hello."), "Hello.");
        let long = "a".repeat(400);
        let name = marker_name(&long);
        assert_eq!(name.chars().count(), MAX_NAME_CHARS);
        assert!(name.ends_with('…'));
    }

    #[test]
    fn lengths_parts_and_estimates_read_in_minutes_and_dollars() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(151.4), "2:31");
        assert_eq!(clock(3_725.0), "1:02:05");
        assert_eq!(clock(f64::NAN), "0:00");
        assert_eq!(parts_for(30.0), 1);
        assert_eq!(parts_for(61.5), 1, "a short rest goes in the last part");
        assert_eq!(parts_for(125.0), 3);
        assert_eq!(parts_for(MAX_TRANSCRIBE_SECONDS), 15);
        assert!(transcribe_estimate(151.0).contains("about $0.006 a minute"));
        assert!(transcribe_estimate(151.0).contains("costs about $0.02."), "{}", transcribe_estimate(151.0));
        assert!(transcribe_estimate(20.0).contains("costs about $0.0020."), "{}", transcribe_estimate(20.0));
        assert!(speech_estimate(412).contains("about $0.015 for 1,000 characters"));
        assert!(speech_estimate(412).contains("costs about $0.0062."), "{}", speech_estimate(412));
    }

    #[test]
    fn audio_files_are_named_by_their_bytes() {
        assert_eq!(audio_ext(b"RIFF....WAVE"), "wav");
        assert_eq!(audio_ext(b"ID3\x04"), "mp3");
        assert_eq!(audio_ext(b"OggS"), "ogg");
        assert_eq!(audio_ext(b""), "mp3");
    }

    #[test]
    fn a_cut_falls_in_the_quietest_moment() {
        let mut x = vec![0.5f32; 1_000];
        for v in &mut x[700..720] {
            *v = 0.0;
        }
        let cut = quietest(&x, 600, 1_000, 20);
        assert!((700..=720).contains(&cut), "{cut}");
        assert_eq!(quietest(&[], 0, 10, 4), 1, "never past the samples, never zero");
    }

    #[test]
    fn a_minute_of_any_audio_fits_one_request() {
        // white noise: the worst case for FLAC
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let noise: Vec<f32> = (0..SEND_RATE as usize * PART_SECONDS)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 24) as f32 * 1.8 - 0.9
            })
            .collect();
        let bytes = flac(&noise).unwrap();
        assert!(bytes.len() < 2_100_000, "{} bytes", bytes.len());
        let (info, buf) = soundcraft_audio_io::decode(&bytes, Some("flac")).unwrap();
        assert_eq!((info.sample_rate, buf.num_channels(), buf.frames()), (SEND_RATE, 1, noise.len()));
    }

    #[test]
    fn sources_follow_the_selection() {
        let mut app = demo_app();
        app.engine.execute("edit.select", &json!({"start": 0, "end": 0})).unwrap();
        assert_eq!(range_source(app.engine.session()), None, "a point is no range");
        app.engine.execute("edit.select", &json!({"tracks": ["Kick"], "start": {"seconds": 12.0}, "end": {"seconds": 14.5}})).unwrap();
        let r = range_source(app.engine.session()).unwrap();
        assert_eq!(r.clip, None);
        assert_eq!(r.label, "The session mix from 0:12 to 0:15.");
        let s = app.engine.session();
        let (track, clip) = s.tracks.iter().find(|t| t.name == "Kick").map(|t| (t.id, t.clips()[0].clone())).unwrap();
        app.engine.session_mut().edit.selected_clips = vec![clip.id];
        let c = clip_source(app.engine.session()).unwrap();
        assert_eq!((c.clip, c.range), (Some((track, clip.id)), clip.range()));
        assert!(c.label.ends_with("on Kick."), "{}", c.label);
    }

    #[test]
    fn the_audio_renders_to_16_khz_mono_parts() {
        let mut app = demo_app();
        let session = app.engine.session_arc();
        let rate = session.sample_rate;
        let range = Range::new(rate.samples(10.0), rate.samples(12.5));
        let src = Source { clip: None, range, label: String::new() };
        let mut prep = Prep::new(&session, &src, Provider::NotSetUp, 0).unwrap();
        render_all(&mut prep);
        assert_eq!(prep.parts.len(), 1);
        let p = &prep.parts[0];
        assert!((p.seconds - 2.5).abs() < 1e-6, "{}", p.seconds);
        assert!((p.samples.len() as f64 - 2.5 * f64::from(SEND_RATE)).abs() < 2.0, "{}", p.samples.len());
        assert!(p.samples.iter().any(|v| v.abs() > 1e-4), "the demo mix is not silent there");
        // the clip alone, by the clip render
        let kick = app.engine.session().tracks.iter().find(|t| t.name == "Kick").unwrap().clips()[0].id;
        app.engine.session_mut().edit.selected_clips = vec![kick];
        let mut src = clip_source(app.engine.session()).unwrap();
        src.range.end = src.range.start + rate.samples(1.5);
        let mut prep = Prep::new(&app.engine.session_arc(), &src, Provider::NotSetUp, 0).unwrap();
        render_all(&mut prep);
        assert_eq!(prep.parts.len(), 1);
        assert!(prep.parts[0].samples.iter().any(|v| v.abs() > 1e-4));
    }

    #[test]
    fn long_audio_goes_out_in_parts_cut_at_quiet_moments() {
        let app = demo_app();
        let session = app.engine.session_arc();
        let rate = session.sample_rate;
        let range = Range::new(0, rate.samples(9.0));
        let src = Source { clip: None, range, label: String::new() };
        let mut prep = Prep::new(&session, &src, Provider::NotSetUp, 0).unwrap();
        // two-second parts, a second of search, half a second of tail
        let hz = rate.hz() as usize;
        (prep.part_len, prep.search_len, prep.tail_len) = (2 * hz, hz, hz / 2);
        render_all(&mut prep);
        assert!(prep.parts.len() >= 4, "{}", prep.parts.len());
        let mut at = 0.0;
        for p in &prep.parts {
            assert!((p.offset - at).abs() < 1e-9, "parts follow each other: {} {at}", p.offset);
            assert!(p.seconds <= 2.5 + 1e-9, "{}", p.seconds);
            at += p.seconds;
        }
        assert!((at - 9.0).abs() < 1e-6, "every sample goes out once: {at}");
    }

    #[test]
    fn silence_is_not_sent_and_a_closed_session_stops_the_audio() {
        assert!(silent(&[0.0; 100]));
        assert!(silent(&[]));
        assert!(!silent(&[0.0, 0.5]));
        let mut app = demo_app();
        with_ai(&mut app, "prep-closed");
        let session = app.engine.session_arc();
        let src = Source { clip: None, range: Range::new(0, 48_000), label: String::new() };
        app.ai.as_mut().unwrap().prep = Some(Prep::new(&session, &src, Provider::NotSetUp, 0).unwrap());
        app.run("session.new", json!({})).unwrap();
        frame(&mut app, &egui::Context::default());
        let ai = app.ai.as_ref().unwrap();
        assert!(ai.prep.is_none() && !ai.working());
        assert!(ai.transcribe.note.as_deref().unwrap_or("").contains("Nothing was sent"));
    }

    #[test]
    fn markers_go_in_at_their_times_in_one_undo_step() {
        let mut app = demo_app();
        with_ai(&mut app, "markers");
        let before = app.engine.session().markers.len();
        let undo = app.engine.undo_history().len();
        let t = Transcript {
            serial: 0,
            start: 2.0,
            segments: vec![Segment { at: 0.0, text: "First line.".into() }, Segment { at: 61.5, text: "Second line.".into() }],
            text: "First line. Second line.".into(),
            done: 70.0,
            length: 70.0,
            stopped: None,
        };
        let note = place_markers(&mut app, &t).unwrap();
        assert!(note.starts_with("2 markers"), "{note}");
        let s = app.engine.session();
        assert_eq!(s.markers.len(), before + 2);
        let rate = s.sample_rate;
        let second = s.markers.iter().find(|m| m.name == "Second line.").unwrap();
        assert_eq!(second.start, rate.samples(63.5));
        assert!(s.markers.iter().any(|m| m.name == "First line." && m.start == rate.samples(2.0)));
        assert_eq!(app.engine.undo_history().len(), undo + 1, "one undo step");
        assert!(app.engine.undo());
        assert_eq!(app.engine.session().markers.len(), before);
    }

    #[test]
    fn markers_wait_while_their_session_is_closed_or_playback_runs() {
        let mut app = demo_app();
        with_ai(&mut app, "markers-wait");
        let t = Transcript {
            serial: 0,
            start: 0.0,
            segments: vec![Segment { at: 1.0, text: "x".into() }],
            text: "x".into(),
            done: 2.0,
            length: 2.0,
            stopped: None,
        };
        app.run("transport.play", json!({})).unwrap();
        app.handle_transport(0.1);
        arrived_transcript(&mut app, t.clone());
        let ai = app.ai.as_ref().unwrap();
        assert_eq!(ai.transcribe.waiting, Some(t.clone()));
        assert!(ai.transcribe.note.as_deref().unwrap_or("").contains("Stop playback first."));
        assert!(app.ui.status.contains("Place Markers"), "the window is closed: the status line says where they wait");
        // a new session: the markers never go there
        app.run("session.new", json!({})).unwrap();
        frame(&mut app, &egui::Context::default());
        app.handle_transport(0.1);
        place_waiting_markers(&mut app);
        let ai = app.ai.as_ref().unwrap();
        assert!(ai.transcribe.note.as_deref().unwrap_or("").contains("is closed"), "{:?}", ai.transcribe.note);
        assert!(app.engine.session().markers.is_empty());
    }

    #[test]
    fn speech_goes_on_a_free_track_at_the_edit_cursor_in_one_undo_step() {
        let mut app = demo_app();
        let dir = with_ai(&mut app, "speech");
        let point = |app: &mut SoundApp, at: Samples| {
            let ed = &mut app.engine.session_mut().edit;
            ed.selected_tracks.clear();
            ed.selection = Range::point(at);
        };
        point(&mut app, 0);
        let bytes = tone_wav(0.5, 24_000);
        let mut v = SpeechOut { ext: "wav", shape: Some((12_000, 24_000, 1)), bytes, path: None };
        let tracks = app.engine.session().tracks.len();
        let undo = app.engine.undo_history().len();
        // Kick (mono) has no clip before bar 5
        let kick = app.engine.session().tracks.iter().find(|t| t.name == "Kick").unwrap().id;
        assert_eq!(free_track(app.engine.session(), 0, 24_000, 1), Some(kick));
        let clips = app.engine.session().track(kick).unwrap().clips().len();
        let note = place_speech(&mut app, &mut v).unwrap();
        assert!(note.contains("on Kick") && note.contains("AI Speech 1.wav"), "{note}");
        assert!(Path::new(v.path.as_deref().unwrap()).starts_with(&dir), "saved in the session's autosave folder");
        let s = app.engine.session();
        assert_eq!(s.tracks.len(), tracks);
        let kick_clips = s.track(kick).unwrap().clips();
        assert_eq!(kick_clips.len(), clips + 1);
        assert!(kick_clips.iter().any(|c| c.start == 0));
        assert_eq!(app.engine.undo_history().len(), undo + 1, "one undo step");
        // no free track at the cursor: a new one
        let at = app.engine.session().sample_rate.samples(30.0);
        point(&mut app, at);
        let mut v2 = SpeechOut { ext: "wav", shape: Some((12_000, 24_000, 1)), bytes: tone_wav(0.5, 24_000), path: None };
        let note = place_speech(&mut app, &mut v2).unwrap();
        assert!(note.contains("new track") && note.contains("AI Speech 2.wav"), "{note}");
        assert_eq!(app.engine.session().tracks.len(), tracks + 1);
    }

    #[test]
    fn speech_waits_while_playback_runs() {
        let mut app = demo_app();
        with_ai(&mut app, "speech-wait");
        app.run("transport.play", json!({})).unwrap();
        app.handle_transport(0.1);
        let v = SpeechOut { bytes: vec![1, 2, 3], ext: "mp3", shape: None, path: None };
        arrived_speech(&mut app, v.clone());
        let ai = app.ai.as_ref().unwrap();
        assert_eq!(ai.speech.waiting, Some(v));
        assert!(ai.speech.note.as_deref().unwrap_or("").contains("Stop playback first."));
    }

    #[test]
    fn the_menu_has_ai_items_that_only_work_where_ai_is_on() {
        use crate::menus;
        let tree = menus::tree();
        let clip = tree.iter().find(|n| n.label == "Clip").unwrap();
        let ai_menu = clip.children.iter().find(|n| n.label == "AI").unwrap();
        let labels: Vec<&str> = ai_menu.children.iter().map(|n| n.label.as_str()).collect();
        assert_eq!(labels, ["Transcribe to Markers...", "Text to Speech Clip...", "AI Settings..."]);
        let extra = menus::ui_aliases();
        for (path, id) in [(TRANSCRIBE_PATH, TRANSCRIBE), (SPEECH_PATH, SPEECH), (SETTINGS_PATH, PREFERENCES), (PREFERENCES_PATH, PREFERENCES)] {
            assert_eq!(menus::command_for_path(path, &extra).as_deref(), Some(id), "{path}");
        }
        let mut app = demo_app();
        assert!(!menus::item_state(&app, TRANSCRIBE_PATH, Some(TRANSCRIBE)).0, "AI is off");
        assert_eq!(app.run(TRANSCRIBE, json!({"value": true})), Err(OFF.into()));
        with_ai(&mut app, "menu");
        assert!(menus::item_state(&app, TRANSCRIBE_PATH, Some(TRANSCRIBE)).0);
        // the items open windows; nothing is sent
        menus::invoke_menu(&mut app, TRANSCRIBE, TRANSCRIBE_PATH);
        menus::invoke_menu(&mut app, SPEECH, SPEECH_PATH);
        menus::invoke_menu(&mut app, PREFERENCES, PREFERENCES_PATH);
        let ai = app.ai.as_ref().unwrap();
        assert!(ai.transcribe_window && ai.speech_window && ai.prefs_window && !ai.working());
    }

    #[test]
    fn the_first_launch_asks_once_and_never_under_control() {
        let ctx = egui::Context::default();
        let mut app = demo_app();
        with_ai(&mut app, "first-run");
        for _ in 0..5 {
            frame(&mut app, &ctx);
        }
        let ai = app.ai.as_ref().unwrap();
        // a credential for the test slug may exist or not: the setup window opens when one is owed
        assert_eq!(ai.setup_window, ai.state.wants_first_run() || ai.state.owes_first_card());
        let mut app = demo_app();
        with_ai(&mut app, "first-run-control");
        app.ai.as_mut().unwrap().skip_first_run();
        for _ in 0..5 {
            frame(&mut app, &ctx);
        }
        assert!(!app.ai.as_ref().unwrap().setup_window);
    }
}
