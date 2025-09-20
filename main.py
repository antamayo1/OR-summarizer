import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
from io import BytesIO
import time

AI_client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

st.set_page_config(page_title="Operating Room Transcriber", layout="wide")

st.title("🎙️🩺⛑️ Operating Room Transcriber")

@st.cache_data
def load_prompts(filename):
  prompts = {}
  with open(filename, "r") as file:
    lines = file.read().split('\n\n')
    for block in lines:
      if block.strip():
        procedure, details = block.split(":", 1)
        prompts[procedure.strip()] = details.strip()
  return prompts

prompts = load_prompts("obgyn_prompts.txt")

if 'prompts' not in st.session_state:
  st.session_state.prompts = prompts

if 'transcription_text' not in st.session_state:
  st.session_state.transcription_text = ""

if 'dictated_text' not in st.session_state:
  st.session_state.dictated_text = ""

if 'selected_procedure' not in st.session_state:
  st.session_state.selected_procedure = None

uploaded_file = st.file_uploader(
  "Upload audio/video file for transcription",
  type=["mp3", "mp4", "wav", "mov", "m4a"]
)
  
if uploaded_file:

  start = time.time()
  with st.spinner("Processing the input audio... Please wait."):
    if 'audio_bytes' not in st.session_state:
      st.session_state.audio_bytes = uploaded_file.read()

    if 'audio' not in st.session_state:
      st.session_state.audio = AudioSegment.from_file(BytesIO(st.session_state.audio_bytes))

    if 'chunk_size_ms' not in st.session_state:
      st.session_state.chunk_size_ms = 600_000

    if 'audio_length_ms' not in st.session_state:
      st.session_state.audio_length_ms = len(st.session_state.audio)

    if 'chunks' not in st.session_state:
      st.session_state.chunks = [
        st.session_state.audio[i:i+st.session_state.chunk_size_ms]
        for i in range(0, st.session_state.audio_length_ms, st.session_state.chunk_size_ms)
      ]
  st.session_state.length = len(st.session_state.audio) / 60000
  if 'chunking_time' not in st.session_state:
    st.session_state.chunking_time = time.time() - start

  st.session_state.selected_procedure = st.selectbox(
    "Select Procedure or Type to Search:",
    options=list(st.session_state.prompts.keys())
  )

  start = time.time()
  if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
    with st.status("Transcribing via OpenAI Whisper API... Please wait.", expanded=True):
      progress_bar = st.progress(0)
      status_placeholder = st.empty()
      status_placeholder.write(
        f"Transcribing... (0% complete)"
      )
      for idx, chunk in enumerate(st.session_state.chunks):

        chunk_io = BytesIO()
        chunk.export(chunk_io, format="mp3")
        chunk_io.seek(0)
        transcription_response = AI_client.audio.transcriptions.create(
          model="whisper-1",
          file=("chunk.mp3", chunk_io, "audio/mp3")
        )
        st.session_state.transcriptions.append(transcription_response.text)
        percent = int(((idx + 1) / len(st.session_state.chunks)) * 100)
        progress_bar.progress(percent / 100)
        status_placeholder.write(
          f"Transcribing... ({percent}% complete)"
        )
  if 'transcription_time' not in st.session_state:
    st.session_state.transcription_time = time.time() - start

  st.session_state.dictated_text = st.text_area("Dictated operative details (type here):", st.session_state.dictated_text, height=150)

if st.session_state.selected_procedure:
  surgery_details = prompts[st.session_state.selected_procedure]

  prompt = f"""
Procedure: {st.session_state.selected_procedure}

{surgery_details}

Dictated Operative Details: {st.session_state.dictated_text}

Operative Transcription:
{' '.join(st.session_state.transcriptions)}

Generate a concise operative summary incorporating all relevant details above.
"""

  summary_prompt = st.text_area("Review and modify summary prompt if necessary:", prompt, height=400)
  start = time.time()
  if st.button("Generate Summary"):
    cumulative_summary = ""
    chunk_summaries = []
    
    with st.status("Generating progressive summary... Please wait.", expanded=True):
      progress_bar = st.progress(0)
      status_placeholder = st.empty()
      status_placeholder.write(
        f"Processing transcription chunks... (0% complete)"
      )
      
      for idx, transcription_chunk in enumerate(st.session_state.transcriptions):
        if idx == 0:
          prompt = f"""
{summary_prompt}

Current Transcription Segment (1/{len(st.session_state.transcriptions)}):
{transcription_chunk}

Generate a concise summary of this first segment of the operative procedure. Pay special attention to the end of this segment as it may continue in the next part.
"""
        else:
          prompt = f"""
{summary_prompt}

Previous Summary:
{cumulative_summary}

Current Transcription Segment ({idx+1}/{len(st.session_state.transcriptions)}):
{transcription_chunk}

Update and expand the summary by incorporating this new segment. Ensure continuity from the previous summary, especially noting any procedures or details that may have been cut off and continued in this segment.
"""
        
        completion = AI_client.chat.completions.create(
          model="gpt-4-turbo",
          messages=[{"role": "user", "content": prompt}]
        )
        
        chunk_summary = completion.choices[0].message.content
        chunk_summaries.append(chunk_summary)
        cumulative_summary = chunk_summary
        
        percent = int(((idx + 1) / len(st.session_state.transcriptions)) * 100)
        progress_bar.progress(percent / 100)
        status_placeholder.write(
          f"Processing Transcription... ({percent}% complete)"
        )

      status_placeholder.write("Creating final comprehensive summary...")
      final_prompt = f"""
{summary_prompt}

Progressive Summaries from Each Transcription Segment:

{'\n'.join([f"\nSegment {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)])}

Create a final, comprehensive operative summary by synthesizing all the progressive summaries above into one cohesive, well-structured operative report and **CRITICAL** DON'T ADD ANYMORE COMMENTARY.

FORMATTING REQUIREMENTS:
- Use proper medical terminology and maintain technical accuracy
- Structure the report with clear headings using **bold** formatting
- Use bullet points (-) for listing procedures, findings, and complications
- Use *italics* for specific anatomical structures and surgical instruments
- Include estimated blood loss, surgical time, and specimen details when mentioned
- Maintain chronological flow of the surgical procedure
- Ensure clarity and conciseness, avoiding redundancy
- Adhere to professional medical documentation standards
- Use third-person perspective and passive voice where appropriate
- Avoid any first-person references (e.g., "I", "we")
- Ensure the report is suitable for inclusion in the patient's official medical record
- Follow the template below exactly, filling in all sections with available information. If a section is not applicable or not mentioned, state "Not applicable" or "No complications" as appropriate.

REQUIRED SECTIONS:
**PREOPERATIVE DIAGNOSIS:**
**POSTOPERATIVE DIAGNOSIS:**
**PROCEDURE PERFORMED:**
**SURGEON:**
**ANESTHESIA:**

**OPERATIVE TECHNIQUE:**
- *Patient positioning and preparation*
- *Surgical approach and incision*
- *Key procedural steps with anatomical landmarks*
- *Closure technique*

**FINDINGS:**
- *Anatomical findings*
- *Pathological findings*
- *Complications (if any)*

**ESTIMATED BLOOD LOSS:**
**SPECIMENS:**
**COMPLICATIONS:**
**CONDITION:**
"""
      
      final_completion = AI_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.1
      )

      st.session_state.summary_text = final_completion.choices[0].message.content
      st.session_state.chunk_summaries = chunk_summaries
      
      status_placeholder.write("Summary generation complete!")

    st.success("✅ Progressive Summary Generated!")
  if 'summary_time' not in st.session_state:
    st.session_state.summary_time = time.time() - start

with st.container(border=True):
  st.subheader("Summary Output", anchor=False)
  st.write(st.session_state.get('summary_text', 'No summary generated yet.'))