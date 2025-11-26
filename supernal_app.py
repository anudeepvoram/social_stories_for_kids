import streamlit as st
import requests
import os
from groq import Groq
from dotenv import load_dotenv
import base64
import re
import glob
import shutil
from gtts import gTTS



# Load .env when running locally
load_dotenv()

# Load secrets (Streamlit Cloud) OR .env (local)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("hf_token")

# ==============================
# Clients
# ==============================
groq_client = Groq(api_key=GROQ_API_KEY)
HEADERS_HF = {"Authorization": f"Bearer {HF_TOKEN}"}

# ==============================
# Streamlit setup
# ==============================
st.set_page_config(page_title="Supernal Social Story Creator", layout="wide")
st.title("Supernal Social Story Creator for Kids")

# ==============================
# Session State Initialization
# ==============================
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "approved" not in st.session_state:
    st.session_state.approved = []
if "final_generated" not in st.session_state:
    st.session_state.final_generated = False
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# ==============================
# Sidebar - API key check
# ==============================
def _mask_key(k: str | None) -> str:
    if not k:
        return "<missing>"
    s = str(k)
    return s[:4] + "..." + s[-4:] if len(s) > 8 else "*" * len(s)

st.sidebar.header("🔐 API Keys Status")
st.sidebar.write("GROQ_API_KEY:", _mask_key(GROQ_API_KEY))
st.sidebar.write("HF_TOKEN:", _mask_key(HF_TOKEN))

# ==============================
# Safety Notice
# ==============================
st.sidebar.markdown("""
### 👶 Child Safety Information
This application is designed for children and includes:
- ✅ Child-friendly, G-rated content only
- ✅ Positive, gentle emotional themes
- ✅ Safe, appropriate imagery
- ✅ Simple, clear language
- ✅ Content safety filters

⚠️ Adult supervision recommended
""")

# ==============================
# User Inputs
# ==============================
st.markdown("""
### Content Guidelines
Please ensure your inputs are:
- Age-appropriate 
- Free from sensitive/adult themes
- Focused on positive learning
- Related to daily activities
""")

col1, col2 = st.columns(2)
with col1:
    # Do not hardcode a child name; leave blank by default and use a helpful placeholder
    child_name = st.text_input("Child's Name", value="", placeholder="Child's name (optional)")
    # Removed hard age limits: accept any numeric age value and default to 6
    child_age = st.number_input("Child's Age", value=6)

    # Add help text for age appropriateness (generic)
    st.info("👶 Content will be tailored to the provided age when available")
    
with col2:
    scenario = st.text_input(
        "Describe the daily situation:",
        "Traveling in a car",
        help="Examples: going to school, sharing toys, meeting new friends"
    )
    traits = st.text_area(
        "Child's Behavior Traits",
        "e.g., gets irritated, avoids talking, poor social attachment",
        help="Focus on behaviors you'd like to improve through positive stories"
    )

num_scenes = st.number_input("Number of scenes to generate", min_value=2, max_value=6, value=3)
image_style = st.selectbox("Choose Image Style:", ["Cartoon", "Animation", "3D Style", "Simple Drawing", "Realistic"])
language_choice = st.selectbox(
    "Choose voice language:",
    ["en", "hi", "te"],
    format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x]
)

# ==============================
# Story Generation (Groq)
# ==============================
def check_content_safety(text: str) -> tuple[bool, str]:
    """Context-aware content safety checker.

    Rules summary:
    - Block sexual/explicit content and graphic violence (blood, gore, mutilation).
    - Block violent actions (shoot, stab, kill, rape, beat) regardless of nearby nouns.
    - Allow neutral mentions of objects (e.g., "gun") when the surrounding context makes clear it's non-violent ("holding a toy gun", "posing with a plastic gun").
    - Block any scene or prompt that describes blood, gore, or graphic injury.
    - Block hateful or explicitly adult content.
    - Allow short, non-empty prompts (>= 3 words).
    """
    text_lower = (text or "").lower()

    # Immediate hard-block tokens (explicitly disallowed)
    hard_block = [
        "porn", "rape", "incest", "sex", "nude", "naked",
        "blood", "gore", "mutilate", "murder", "suicide"
    ]
    for token in hard_block:
        if re.search(rf"\b{re.escape(token)}\b", text_lower):
            return False, f"Found explicit/graphic content: {token}"

    # Violent action verbs - disallow if present
    violent_verbs = ["shoot", "shooting", "stab", "stabbed", "kill", "killed", "murder", "attack", "beat", "beat up", "punch", "hit", "bleed", "bleeding", "strangle", "choke"]
    for v in violent_verbs:
        if v in text_lower:
            return False, f"Found violent action: {v}"

    # Hate/abuse indicators - simple heuristic
    hate_indicators = ["hate", "racist", "slur", "kill them", "exterminate"]
    for h in hate_indicators:
        if h in text_lower:
            return False, f"Found hateful content indicator: {h}"

    # Profanity pattern (catch common obfuscations)
    profanity_pattern = re.compile(r"(f[\*\w]*ck|sh[\*\w]*t|b[\*\w]*tch|d[\*\w]*mn|h[\*\w]*ll)")
    if profanity_pattern.search(text_lower):
        return False, "Found profanity"

    # Weapon mentions: allow when clearly non-violent (toy/pretend/play/holding/posing) or when no violent verbs appear in the whole text
    weapon_terms = ["gun", "knife", "weapon", "blade", "rifle", "pistol"]
    safe_context_markers = ["toy", "pretend", "play", "plastic", "pretend-play", "imaginary", "holding", "posing", "holster", "decorative"]
    has_violent_verbs_in_text = any(v in text_lower for v in violent_verbs)
    
    for w in weapon_terms:
        for m in re.finditer(rf"\b{w}\b", text_lower):
            start = max(0, m.start() - 80)
            end = min(len(text_lower), m.end() + 80)
            context = text_lower[start:end]
            
            # If any explicit graphic token appears near the weapon, block
            if any(h in context for h in hard_block):
                return False, f"Weapon used in graphic context: {w}"
            
            # If safe markers are present in context, allow
            if any(marker in context for marker in safe_context_markers):
                continue
            
            # If no violent verbs anywhere in the text, allow neutral weapon mentions (safer heuristic)
            if not has_violent_verbs_in_text:
                continue
            
            # If violent verbs exist in the text AND near this weapon mention, block
            if any(v in context for v in violent_verbs):
                return False, f"Weapon used in violent context: {w}"

    # Length checks
    words = [w for w in re.findall(r"\w+", text_lower)]
    if len(words) < 3:
        return False, "Response too short (minimum 3 words)"

    # Passed safety heuristics
    return True, ""

def generate_story(prompt: str) -> str:
    """Generate a story with enhanced safety checks."""
    # Add explicit safety constraints to the prompt
    safety_prefix = """
    CRITICAL SAFETY REQUIREMENTS:
    1. Generate ONLY child-friendly, G-rated content suitable for ages 3-12
    2. NO adult themes, violence, scary situations, or inappropriate content
    3. NO swearing, rude words, or offensive language
    4. Keep emotional content gentle and positive
    5. Focus on learning, growth, and positive social interactions
    6. Avoid references to danger, injury, or distressing situations
    7. Use simple, clear language appropriate for young children
    8. Keep each scene concise (2-3 sentences, about 50-100 words per scene)
    
    Remember: This content is for young children's social stories.
    Format each scene as:
    Scene X:
    Scene Text: [2-3 concise sentences]
    """
    
    safe_prompt = safety_prefix + "\n" + prompt
    
    # Generate content with retries and progressive backoff
    max_attempts = 5  # Increased from 3 to 5 attempts
    backoff_seconds = [0, 1, 2, 4, 8]  # Progressive backoff
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            # Add attempt number to help vary the output
            attempt_prompt = f"{safe_prompt}\nAttempt: {attempt + 1}"
            
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a children's storyteller creating short, clear scenes."},
                    {"role": "user", "content": attempt_prompt}
                ]
            )
            content = response.choices[0].message.content
            
            # Check content safety
            is_safe, reason = check_content_safety(content)
            if is_safe:
                # Return content and let the caller handle formatting/parsing
                return content
            else:
                last_error = reason
            
            # If we get here, content wasn't safe or properly formatted
            if attempt < max_attempts - 1:
                import time
                time.sleep(backoff_seconds[attempt])  # Progressive backoff
                continue
                
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts - 1:
                import time
                time.sleep(backoff_seconds[attempt])
                continue
    
    error_msg = f"Story generation failed after {max_attempts} attempts: {last_error}"
    raise ValueError(error_msg)


def detect_hallucination(text: str, scenario: str, allowed_names: list[str] | None = None) -> tuple[bool, str]:
    """Detect simple hallucinations: mentions of relatives/locations/people not present in the given scenario.

    This is a conservative heuristic to catch common hallucinations like "grandparents", "house/home",
    "school", "teacher", "friends" etc. If one of these tokens appears in the text but not in the
    scenario (and not part of allowed_names), it is flagged as a hallucination.

    Returns (is_hallucinated, reason token)
    """
    text_lower = (text or "").lower()
    scenario_lower = (scenario or "").lower()
    allowed = " ".join(allowed_names or []).lower()

    hallucination_indicators = [
        "grandparent", "grandparents", "grandma", "grandpa", "house", "home",
        "school", "teacher", "park", "friend", "friends", "store", "restaurant",
        "birthday", "party", "sibling", "brother", "sister", "neighbor", "aunt", "uncle"
    ]

    for token in hallucination_indicators:
        if token in text_lower:
            # If the scenario explicitly mentioned it, it's allowed
            if token in scenario_lower:
                continue
            # If allowed names string contains the token (rare), allow
            if token in allowed:
                continue
            return True, token

    return False, ""


def remove_hallucinated_sentences(text: str, tokens: list[str]) -> str:
    """Remove sentences that contain any of the given tokens. Simple fallback when regeneration fails."""
    import re
    # Split on sentence endings while preserving content
    sentences = re.split(r'(?<=[.!?])\s+', text)
    tokens_lower = [t.lower() for t in tokens]
    kept = []
    for s in sentences:
        s_lower = s.lower()
        if any(t in s_lower for t in tokens_lower):
            # drop this sentence
            continue
        kept.append(s)
    return " ".join(kept).strip()

# ==============================
# Image Generation (Hugging Face)
# ==============================
def generate_detailed_image(scene_prompt, style, child_name, child_age, traits, scenario):
    # First check if the scene prompt is safe
    is_safe, reason = check_content_safety(scene_prompt)
    if not is_safe:
        raise ValueError(f"Scene content failed safety check: {reason}")
    
    style_map = {
        "Cartoon": (
            "masterful Pixar-quality 3D animation, award-winning animation style, "
            "stunning detailed textures, volumetric lighting, ambient occlusion, "
            "subsurface scattering on skin, ray-traced reflections, "
            "cinematic depth of field, high-end CGI rendering, "
            "perfect color grading, movie-quality animation, "
            "intricate facial expressions, fluid character movement, "
            "exceptionally detailed environments, dynamic lighting effects, "
            "professional character design, Pixar/Disney/Dreamworks quality"
        ),
        "Animation": (
            "supreme quality 2D animation, Studio Ghibli level artistry, "
            "masterful hand-drawn aesthetic, exceptional cel-shading, "
            "stunning attention to detail, perfect line quality, "
            "beautiful color composition, award-winning animation style, "
            "incredibly smooth movement, dynamic poses, "
            "expert use of perspective, masterful lighting and shadows, "
            "premium quality backgrounds, impeccable character design, "
            "industry-leading animation techniques, theatrical release quality"
        ),
        "3D Style": (
            "cutting-edge 3D rendering, photorealistic textures, "
            "next-gen global illumination, perfect volumetric lighting, "
            "cinema-quality CGI, high-fidelity material shading, "
            "ultra-detailed geometry, professional rigging, "
            "advanced particle effects, ray-traced shadows, "
            "state-of-the-art animation, movie-quality rendering, "
            "exceptional facial animation, fluid cloth simulation, "
            "studio-quality character design, AAA production value"
        ),
        "Simple Drawing": (
            "premium children's book illustration style, "
            "masterful watercolor techniques, expert composition, "
            "perfect color harmony, exceptional artistic detail, "
            "award-winning illustration quality, refined linework, "
            "professional art direction, beautiful texturing, "
            "stunning use of negative space, dynamic composition, "
            "gallery-quality artwork, masterful brushwork, "
            "picture-book perfection, artistic excellence"
        ),
        "Realistic": (
            "ultra high-quality professional photography, award-winning portrait style, masterful composition, "
            "8K UHD, RAW quality, hyperdetailed, masterful photography, perfect lighting, "
            "high-end DSLR, 85mm f/1.4 lens, golden hour natural sunlight, cinematic color grading, "
            "photorealistic textures, stunning detail, professional retouching, "
            "family-friendly setting, wholesome atmosphere, perfect exposure, "
            "crisp focus, beautiful bokeh effect, professional studio quality, "
            "age-appropriate styling, natural expressions, authentic emotions, "
            "crystal clear, vibrant colors, impeccable image quality, magazine quality"
        )
    }

    behavior_hint = (
        f"The child subtly shows {traits}, expressed softly through facial emotion or body language — not the main focus."
        if traits else ""
    )

    final_prompt = (
        f"{style_map[style]}. "
        f"Main focus: {scene_prompt}. "
        f"Illustrate a {child_age}-year-old child named {child_name}. "
        f"{behavior_hint} "
        f"Keep the mood positive, emotionally gentle, and suitable for children's storytelling."
    )

    if style == "Realistic":
        final_prompt += " --photo --ultra-realistic --no cartoon --no painting --human likeness"

    HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    payload = {"inputs": final_prompt, "parameters": {"guidance_scale": 7.5}}

    response = requests.post(HF_API_URL, headers=HEADERS_HF, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Image generation failed: {response.text}")
        return None

# ==============================
# Voice Generation (GTTS)
# ==============================
def generate_voice_gtts(scene_text, language="en", output_file="scene.mp3"):
    try:
        tts = gTTS(text=scene_text, lang=language)
        tts.save(output_file)
        return output_file
    except Exception as e:
        st.error(f"GTTS Error: {e}")
        return None

# ==============================
# Combine GTTS clips (Pure Python)
# ==============================
def combine_gtts_clips(audio_files, output_file="combined_story.mp3"):
    with open(output_file, "wb") as outfile:
        for idx, fname in enumerate(audio_files):
            with open(fname, "rb") as infile:
                data = infile.read()
                if idx == 0:
                    outfile.write(data)
                else:
                    frame_start = data.find(b"\xff\xfb")
                    if frame_start != -1:
                        outfile.write(data[frame_start:])
    return output_file


def generate_image_prompt(scene_text, child_name, child_age, traits, scenario, style):
    """Use the LLM to create a concise image prompt for an approved scene text."""
    prompt = f"""
    You are a visual prompt generator for an image model. Given the following scene description, produce a single concise image prompt (one or two short sentences) suitable for an image generation API. Include the child's age and name, setting, characters' expressions and key objects. Do NOT include camera jargon or special tokens.

    Scene description:
    {scene_text}

    Child name: {child_name}
    Child age: {child_age}
    Situation: {scenario}
    Traits: {traits}

    Return only the image prompt text.
    """
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        generated = resp.choices[0].message.content.strip()
        # Validate generated prompt for safety before returning
        is_safe, reason = check_content_safety(generated)
        if not is_safe:
            # If the generated prompt is unsafe, fall back to the original scene text (safer)
            return scene_text
        return generated
    except Exception as e:
        # Fallback: use the scene text as a simple prompt
        return scene_text

# helper: clear generated files
def clear_generated_files():
    patterns = ["scene_*.png", "preview_scene_*.png", "scene_*.mp3", "combined_story.mp3"]
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except Exception:
                pass
    st.session_state.generated_images = []
    st.session_state.final_generated = False

# ==============================
# Step 1: Generate editable scenes (LLM only)
# ==============================
def _generate_scenes_from_llm(scenario_input, traits_input, child_name_input, child_age_input, num_scenes_input, is_regeneration=False):
    """Generate scenes from LLM. Takes inputs as parameters to ensure current UI values are used."""
    import random
    import time

    # Respect multiple comma-separated situations if provided
    if not scenario_input.strip():
        st.warning("⚠️ Please enter at least one situation.")
        return

    # Random elements to help force variation
    time_settings = ["morning", "afternoon", "evening", "after lunch", "before dinner"]
    emotions = ["excited", "curious", "calm", "happy", "interested", "enthusiastic"]
    activities = ["playing", "learning", "exploring", "practicing", "trying", "discovering"]
    
    scenario_parts = [s.strip() for s in scenario_input.split(",") if s.strip()]
    situation_instruction = ""
    if len(scenario_parts) > 1:
        situation_instruction = (
            f"IMPORTANT: This story must cover ALL these situations across the {num_scenes_input} scenes:\n"
            + "\n".join(f"- {s}" for s in scenario_parts)
            + "\nMake sure EACH scene focuses on ONE of these situations."
        )
    
    # Add variation instruction when regenerating
    variation_instruction = ""
    if is_regeneration:
        # Add random elements to force story variation
        random_time = random.choice(time_settings)
        random_emotion = random.choice(emotions)
        random_activity = random.choice(activities)
        
        variation_instruction = f"""
        CRITICAL: Generate a COMPLETELY NEW and DIFFERENT story:
        - Set during {random_time}
        - {child_name_input} should be {random_emotion}
        - Include {random_activity} as a key activity
        - Use entirely different scenes and actions
        - Change the sequence of events
        - Explore a different approach to the situation
        - Do NOT reuse any elements from previous stories
        
        THIS MUST BE A COMPLETELY FRESH STORY WITH NO SIMILARITY TO PREVIOUS VERSIONS.
        """
    
    # Add timestamp to make each request unique
    unique_seed = str(time.time())

    prompt = f"""
    You are a children's storyteller. Write a focused, realistic story in {num_scenes_input} scenes.
    
    STRICT CONTENT RULES:
    1. Focus ONLY on the exact situation provided: {scenario_input}
    2. Address ONLY the specific behavioral traits: {traits_input}
    3. DO NOT add locations, people, or events that weren't mentioned
    4. DO NOT invent additional story elements or backgrounds
    5. Keep each scene focused on managing the specific behavior in the given situation
    6. Use realistic, practical responses to the behavior
    7. Stay in the present moment of the situation
    
    Child: {child_name_input}, age {child_age_input}
    Daily situation (STICK TO THIS ONLY): {scenario_input}
    Behavioral traits to address: {traits_input}
    {situation_instruction}
    {variation_instruction}
    
    Unique Request ID: {unique_seed}
    
    Follow this EXACT format for EACH scene:
    Scene 1:
    Scene Text: [2-3 sentences focused strictly on the given situation and specific behaviors]
    
    Scene 2:
    Scene Text: [2-3 sentences showing how to handle the exact behaviors in this specific situation]
    
    Note: Each scene must ONLY describe the exact situation ({scenario_input}) and behaviors ({traits_input}).
    
    DO NOT include:
    - Made up destinations
    - Additional family members
    - Imagined activities
    - Future events
    - Past events
    - Any details not directly related to the car situation and behavior
    
    Keep focus on:
    - The immediate car situation
    - The specific behavioral challenges
    - Practical coping strategies
    - Real-time solutions
    """

    # Generate story and guard against hallucinations (mentions of relatives/places not in `scenario_input`)
    max_hallucination_attempts = 3
    story_text = None
    last_halluc_reason = ""
    prompt_with_halluc_block = prompt
    
    try:
        for h_attempt in range(max_hallucination_attempts):
            try:
                story_text = generate_story(prompt_with_halluc_block)
                is_halluc, token = detect_hallucination(story_text, scenario_input, allowed_names=[child_name_input])
                if not is_halluc:
                    break
                # If hallucination detected, add an explicit prohibition and retry
                last_halluc_reason = token
                prohibition = (
                    f"\nCRITICAL: DO NOT mention any relatives, locations, or people that were not provided in the 'Daily situation' input. "
                    f"Specifically, do NOT add the word '{token}' or related concepts unless they were explicitly listed in the scenario."
                )
                prompt_with_halluc_block = prompt + "\n" + prohibition
            except ValueError as e:
                # If story generation fails, warn user and allow retry
                if h_attempt < max_hallucination_attempts - 1:
                    st.warning(f"⚠️ Story generation attempt {h_attempt + 1} had issues. Retrying with adjusted safety settings...")
                    import time
                    time.sleep(1)
                    continue
                else:
                    raise

        # If still hallucinated after retries, try stripping offending sentences as a conservative fallback
        is_halluc, token = detect_hallucination(story_text or "", scenario_input, allowed_names=[child_name_input])
        if is_halluc:
            # collect all tokens that appear
            detected = [t for t in [
                "grandparent", "grandparents", "grandma", "grandpa", "house", "home",
                "school", "teacher", "park", "friend", "friends", "store", "restaurant",
                "birthday", "party", "sibling", "brother", "sister", "neighbor", "aunt", "uncle"
            ] if t in (story_text or "").lower() and t not in scenario_input.lower()]
            if detected:
                story_text = remove_hallucinated_sentences(story_text or "", detected)
                st.info(f"ℹ️ Removed hallucinated content for cleaner focus on the scenario.")
        
        # First try to extract with strict format
        scene_texts = re.findall(r"Scene\s*\d+[:\-]*\s*Scene Text:\s*(.*?)(?=Scene\s*\d+|$)", story_text, flags=re.S)
        
        if not scene_texts:
            # Fallback: try simpler format (Scene 1: text)
            scene_texts = re.findall(r"Scene\s*\d+[:\-]\s*(.*?)(?=Scene\s*\d+|$)", story_text, flags=re.S)
        
        if not scene_texts:
            # Last resort: just split by newlines and look for non-empty lines
            scene_texts = [s.strip() for s in story_text.split("\n") if s.strip() and not s.lower().startswith("scene")]
        
        st.session_state.scenes = [s.strip() for s in scene_texts][:num_scenes_input]
        if not st.session_state.scenes:
            st.error("⚠️ Could not extract scenes from the story. Please try regenerating.")
    
    except Exception as e:
        st.error(f"⚠️ Story generation encountered an issue: {str(e)}\n\nPlease try again or adjust your inputs for a simpler scenario.")
    
    st.session_state.approved = [False] * len(st.session_state.scenes)
    st.session_state.final_generated = False

# Generate or regenerate scenes
if st.button("📝 Generate Editable Scenes"):
    clear_generated_files()
    _generate_scenes_from_llm(scenario, traits, child_name, child_age, num_scenes, is_regeneration=False)

if st.button("🔁 Regenerate Scenes"):
    # allow user to regenerate if they want a different story
    clear_generated_files()
    _generate_scenes_from_llm(scenario, traits, child_name, child_age, num_scenes, is_regeneration=True)

# ==============================
# Step 2: Approve scenes (user edits scene text only)
# ==============================
for idx, scene in enumerate(st.session_state.scenes):
    st.markdown(f"## Scene {idx+1}")
    if not st.session_state.approved[idx]:
        edited_scene = st.text_area(f"Edit Scene {idx+1}", value=scene, key=f"scene_{idx}")
        if st.button(f"✅ Approve Scene {idx+1}", key=f"approve_{idx}"):
            # save edited scene
            st.session_state.scenes[idx] = edited_scene
            st.session_state.approved[idx] = True

            # generate a preview image immediately for the approved scene so user sees it below the approved block
            preview_path = f"preview_scene_{idx+1}.png"
            try:
                with st.spinner(f"🎨 Generating preview for Scene {idx+1}..."):
                    img_prompt = generate_image_prompt(edited_scene, child_name, child_age, traits, scenario, image_style)
                    img = generate_detailed_image(img_prompt, image_style, child_name, child_age, traits, scenario)
                    if img:
                        with open(preview_path, "wb") as f:
                            f.write(img)
            except Exception:
                # ignore preview failures but continue
                pass

            st.success(f"Scene {idx+1} approved!")
    else:
        st.markdown(f"**{scene}**")
        st.success("✅ Approved")
        # show preview image under the approved scene if available
        preview_path = f"preview_scene_{idx+1}.png"
        if os.path.exists(preview_path):
            st.image(preview_path, caption=f"Preview (Scene {idx+1})", use_column_width=True)
# ==============================
# Step 3: Generate final images + voices + slideshow
# ==============================
if st.session_state.approved and all(st.session_state.approved) and not st.session_state.final_generated:
    st.session_state.generated_images = []
    audio_files = []

    for idx, scene in enumerate(st.session_state.scenes):
        st.markdown(f"## 🖼 Scene {idx+1}")
        with st.spinner(f"🎨 Generating image for Scene {idx+1}..."):
            # Generate an image prompt from the approved scene text, then generate the image
            final_img_path = f"scene_{idx+1}.png"
            try:
                image_prompt = generate_image_prompt(
                    scene, child_name, child_age, traits, scenario, image_style
                )
            except Exception:
                image_prompt = scene

            # Prefer any preview generated at approval time
            preview_path = f"preview_scene_{idx+1}.png"
            if os.path.exists(preview_path):
                try:
                    shutil.copyfile(preview_path, final_img_path)
                    st.session_state.generated_images.append(final_img_path)
                except Exception:
                    # fallback to generating anew
                    img = generate_detailed_image(
                        image_prompt,
                        image_style,
                        child_name,
                        child_age,
                        traits,
                        scenario
                    )
                    if img:
                        with open(final_img_path, "wb") as f:
                            f.write(img)
                        st.session_state.generated_images.append(final_img_path)
            else:
                img = generate_detailed_image(
                    image_prompt,
                    image_style,
                    child_name,
                    child_age,
                    traits,
                    scenario
                )
                if img:
                    with open(final_img_path, "wb") as f:
                        f.write(img)
                    st.session_state.generated_images.append(final_img_path)

        with st.spinner(f"🔊 Generating voice for Scene {idx+1}..."):
            audio_file = f"scene_{idx+1}.mp3"
            audio_path = generate_voice_gtts(scene, language_choice, audio_file)
            if audio_path:
                audio_files.append(audio_path)

    combined_output = combine_gtts_clips(audio_files)
    st.success("✅ All scenes generated successfully!")

    if st.session_state.generated_images:
        st.markdown("### 🎞️ Cinematic Story Slideshow")

        try:
            from mutagen.mp3 import MP3
            audio = MP3(combined_output)
            total_duration = audio.info.length
            per_scene_duration = total_duration / len(st.session_state.generated_images)
        except Exception:
            per_scene_duration = 15

        # Show the full story as a paragraph
        combined_text = "\n\n".join(st.session_state.scenes)
        st.markdown("### 📖 Full story")
        st.write(combined_text)

        # Build JS-synced slideshow that follows the audio playback
        slides_html = ""
        for i, img_path in enumerate(st.session_state.generated_images):
            with open(img_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                slides_html += f'\n<div class="slide" id="slide-{i}">\n  <img src="data:image/png;base64,{data}" class="zoom-image"/>\n</div>\n'

        # Read combined audio and embed as base64 so we can control playback from the same HTML
        audio_b64 = ""
        try:
            with open(combined_output, "rb") as af:
                audio_b64 = base64.b64encode(af.read()).decode()
        except Exception:
            audio_b64 = ""

        num_slides = len(st.session_state.generated_images)
        per_scene = per_scene_duration

        slideshow_html = f"""
        <style>
        .slideshow-container {{
            max-width: 750px;
            height: 420px;
            position: relative;
            margin: auto;
            overflow: hidden;
            border-radius: 20px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        .slide {{
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            transition: opacity 0.6s ease-in-out, transform 0.6s ease-in-out;
            transform: scale(1.02);
        }}
        .slide.active {{
            opacity: 1;
            transform: scale(1);
        }}
        .zoom-image {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        #scene-indicator {{ text-align: center; margin-top: 10px; font-size: 1.1em; }}
        </style>

        <div class="slideshow-container">
            {slides_html}
        </div>
        <div id="scene-indicator">Current Scene: <span id="current-scene">1</span></div>
        <audio id="narration" controls preload="auto">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>

        <script>
        const numSlides = {num_slides};
        const perScene = {per_scene};
        function setActive(idx) {{
            for (let i=0;i<numSlides;i++) {{
                const el = document.getElementById('slide-' + i);
                if (!el) continue;
                if (i === idx) el.classList.add('active'); else el.classList.remove('active');
            }}
            document.getElementById('current-scene').textContent = Math.min(idx+1, numSlides);
        }}
        const audio = document.getElementById('narration');
        audio.addEventListener('timeupdate', function() {{
            const t = audio.currentTime;
            let idx = Math.floor(t / perScene);
            if (idx < 0) idx = 0;
            if (idx >= numSlides) idx = numSlides - 1;
            setActive(idx);
        }});
        // Initialize first slide
        setActive(0);
        </script>
        """

        st.components.v1.html(slideshow_html, height=620)

    st.session_state.final_generated = True

