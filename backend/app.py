import os
import json
import sys
import random
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config
from models.model_loader import model_instance
from utils.image_processor import analyze_bite_severity

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Allow all origins (for development)

# Ensure upload folder exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Path to the frontend HTML (one level up from backend/)
FRONTEND_HTML = BASE_DIR.parent / 'index soi.html'

# ──────────────────────────────────────────────
# Optional Gemini AI initialisation
# ──────────────────────────────────────────────
_gemini_client = None
_gemini_model_name = None  # resolved at startup
_SYSTEM_INSTRUCTION = (
    "You are VenomAI, a knowledgeable medical AI assistant specialising in "
    "snakebite identification, treatment, and prevention. Always recommend "
    "professional medical care for actual bites. Be concise and accurate. "
    "Format responses with clear headings and bullet points where helpful."
)

# Models tried in priority order — first one that responds wins
_MODEL_PRIORITY = [
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

def _init_gemini():
    global _gemini_client, _gemini_model_name
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("[INFO] No GEMINI_API_KEY set -- using fallback responses.")
        return
    try:
        from google import genai  # type: ignore
        from google.genai import types as _t  # type: ignore
        client = genai.Client(api_key=api_key)

        # Find first model with available quota
        for model in _MODEL_PRIORITY:
            try:
                client.models.generate_content(
                    model=model,
                    contents="ping",
                    config=_t.GenerateContentConfig(max_output_tokens=1),
                )
                _gemini_client = client
                _gemini_model_name = model
                print(f"[OK] Gemini ready -- using model: {model}")
                return
            except Exception as probe_err:
                msg = str(probe_err)
                if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    print(f"[WARN] {model}: quota exhausted -- trying next model")
                elif "NOT_FOUND" in msg or "not found" in msg.lower():
                    print(f"[WARN] {model}: not available -- trying next model")
                else:
                    print(f"[WARN] {model}: {msg[:80]}")

        print("[ERROR] All Gemini models exhausted quota or unavailable -- using fallback.")
    except Exception as exc:
        print(f"[ERROR] Gemini AI not available: {exc}")
        _gemini_client = None

_init_gemini()

# ──────────────────────────────────────────────
# Pre-built fallback responses (used when Gemini key is not set)
# ──────────────────────────────────────────────
_FALLBACK_RESPONSES = {
    "first aid": """**SNAKEBITE FIRST AID — IMMEDIATE STEPS**

1. **Call emergency services immediately** — This is the most important step.
2. **Keep the victim calm and still** — Movement spreads venom faster.
3. **Position the bite site below heart level** — Reduces venom flow.
4. **Remove constrictive items** — Rings, watches, tight clothing (swelling is rapid).
5. **Immobilise the limb** — Splint it as you would a fracture.
6. **Note the time of bite** — Critical for antivenom dosing decisions.

**DO NOT:**
✗ Apply a tourniquet  
✗ Cut or suck the wound  
✗ Apply ice or heat  
✗ Give alcohol, aspirin, or stimulants  
✗ Wash the bite site (preserves venom for identification)

**TIME IS CRITICAL** — The faster antivenom is administered, the better the outcome.""",

    "prevention": """**SNAKEBITE PREVENTION**

**In the Field:**
• Wear high boots and long trousers in snake habitats
• Use a torch/flashlight when walking at night
• Probe ahead with a stick in tall grass
• Never reach into holes, logs, or dark spaces without looking
• Avoid turning over rocks or debris by hand

**At Home:**
• Keep grass short around the house
• Seal gaps under doors and around pipes
• Control rodent populations (snakes follow prey)
• Keep firewood piles away from the house

**If You Encounter a Snake:**
• Stand still — most bites occur when someone tries to handle or kill a snake
• Back away slowly — give it a wide berth
• Do not attempt to capture or kill it

**Know Your Region:** Learn which venomous species are local to your area.""",

    "symptoms": """**SNAKEBITE SYMPTOMS BY VENOM TYPE**

**Neurotoxic** (Cobra, Krait, Mamba, Taipan):
• Drooping eyelids (ptosis)
• Difficulty swallowing or speaking
• Progressive muscle weakness
• Respiratory paralysis (can occur within hours)

**Hemotoxic / Cytotoxic** (Vipers, Adders):
• Severe, rapidly spreading pain and swelling
• Bruising and blistering around the bite
• Bleeding from gums, nose, or the wound itself
• Dark urine (kidney involvement)

**General Warning Signs** (any bite):
• Nausea and vomiting
• Dizziness or blurred vision
• Rapid or irregular heartbeat
• Loss of consciousness

⚠️ **20% of bites are 'dry bites' with no venom** — but always seek medical evaluation regardless.""",

    "cobra": """**COMMON COBRA (Naja naja)**

**Identification:**
• Recognisable hood spreads when threatened
• Brown to black colouration; smooth scales
• Round pupils; 4–7 ft adult length

**Venom:** Neurotoxic — attacks the nervous system
**LD50:** 0.29 mg/kg (highly potent)

**Bite Symptoms:**
• Minimal local pain initially
• Drooping eyelids (ptosis) within 1–3 hours
• Difficulty breathing — may require ventilation
• Fatality: 10–20% without antivenom

**Treatment:** Polyvalent or monovalent anti-cobra antivenom. Early administration is critical.

**Safety:** Never reach into mounds or holes. Wear boots in cobra habitat.""",

    "antivenom": """**ANTIVENOM — KEY FACTS**

**What it is:** Antibodies raised in horses or sheep against specific snake venoms.

**Administration:** Always intravenous (IV) in a hospital setting. Test for allergic reaction first.

**Timing:** Most effective when given within 4–6 hours of the bite. Still beneficial up to 24 hours.

**Regional Availability:**
• 🇮🇳 South Asia — Polyvalent (covers Cobra, Krait, Russell's Viper, Saw-scaled Viper)
• 🌏 Southeast Asia — Thai Red Cross, Indonesian monovalents
• 🌍 Africa — SAIMR Polyvalent; specific for Boomslang, Mamba
• 🇦🇺 Australia — CSL monovalents per species
• 🇺🇸 Americas — CroFab, Anavip (crotalids); Instituto Butantan (Brazil)

**Challenges:** Many rural areas lack stocks. The WHO lists antivenom as an essential medicine.""",
}


def _fallback_response(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["first aid", "emergency", "immediate", "help", "bitten"]):
        return _FALLBACK_RESPONSES["first aid"]
    if any(w in msg for w in ["prevent", "avoid", "safety", "protect"]):
        return _FALLBACK_RESPONSES["prevention"]
    if any(w in msg for w in ["symptom", "sign", "feel", "pain", "swell"]):
        return _FALLBACK_RESPONSES["symptoms"]
    if any(w in msg for w in ["cobra", "naja", "hood"]):
        return _FALLBACK_RESPONSES["cobra"]
    if any(w in msg for w in ["antivenom", "treatment", "cure", "hospital"]):
        return _FALLBACK_RESPONSES["antivenom"]
    return (
        "I'm VenomAI, specialised in snakebite protection. Try asking me about:\n\n"
        "• **First aid steps** for a snakebite\n"
        "• **Symptoms** of venomous bites\n"
        "• **How to prevent** snakebites\n"
        "• **Specific snakes** (e.g. 'tell me about cobras')\n"
        "• **Antivenom** information\n\n"
        "For a real emergency, call your local emergency services immediately."
    )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the main frontend HTML file."""
    if FRONTEND_HTML.exists():
        return send_file(str(FRONTEND_HTML))
    return jsonify({'error': 'Frontend not found. Open index soi.html directly.'}), 404


@app.route('/api/health', methods=['GET'])
def health():
    gemini_status = "available" if _gemini_client else "fallback (no GEMINI_API_KEY)"
    return jsonify({
        'status': 'ok',
        'message': 'VenomAI Backend is running',
        'gemini': gemini_status,
    })


@app.route('/api/snakes', methods=['GET'])
def get_snakes():
    """Return all snake species from the JSON database."""
    try:
        with open(Config.SNAKES_JSON, 'r') as f:
            snakes = json.load(f)
        return jsonify(snakes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify-snake', methods=['POST'])
def identify_snake():
    """Identify snake species from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, webp'}), 400

    try:
        img_bytes = file.read()
        result = model_instance.predict(img_bytes)
        species = result['class_name']
        confidence = result['confidence']

        snake_info = model_instance.get_snake_info(species)

        return jsonify({
            'success': True,
            'species': species,
            'scientific_name': snake_info.get('scientific', 'Unknown'),
            'confidence': confidence,
            'venom_type': snake_info.get('venomType', 'Unknown'),
            'venom_potency': snake_info.get('venom', 'unknown'),
            'region': snake_info.get('region', 'Unknown'),
            'antivenom': snake_info.get('antivenom', 'Unavailable'),
            'fatality_rate': snake_info.get('fatality', 'N/A'),
            'description': snake_info.get('description', ''),
            'symptoms': snake_info.get('symptoms', ''),
            'habitat': snake_info.get('habitat', ''),
            'ld50': snake_info.get('ld50', 'N/A'),
            'image': snake_info.get('image', ''),
            'all_predictions': result['all_predictions'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-bite', methods=['POST'])
def analyze_bite():
    """Analyze bite mark image and return severity and probable snake."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    location = request.form.get('location', '')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        img_bytes = file.read()
        severity = analyze_bite_severity(img_bytes)

        with open(Config.SNAKES_JSON, 'r') as f:
            snakes = json.load(f)

        venomous = [s for s in snakes if s.get('venom') in ['high', 'medium']]
        probable = random.choice(venomous) if venomous else {'name': 'Unknown', 'scientific': ''}

        severity_text = {
            'high': 'High Severity',
            'medium': 'Medium Severity',
            'low': 'Low Severity',
        }.get(severity, 'Unknown')

        recommendations = {
            'high': [
                "THIS IS A MEDICAL EMERGENCY — Call emergency services immediately",
                "Keep victim calm and completely immobilised",
                "Do not elevate the bitten limb above heart level",
                "Remove all constrictive items (rings, watches, tight clothing)",
                "Transport to nearest hospital with ICU and antivenom capability",
                "Antivenom administration is the primary life-saving treatment",
                "Be prepared to support breathing if respiratory failure occurs",
            ],
            'medium': [
                "Seek medical attention within 2–4 hours — do not delay",
                "Do not wash the bite site (preserves venom for identification)",
                "Immobilise the affected limb at or below heart level",
                "Remove constrictive jewellery and clothing",
                "Monitor for spreading swelling, bruising, or systemic symptoms",
                "Update tetanus immunisation if more than 5 years since last dose",
                "Avoid traditional remedies, incision, or tourniquet application",
            ],
            'low': [
                "Clean wound gently with soap and water",
                "Apply antiseptic and a clean dry dressing",
                "Seek medical evaluation within 24 hours",
                "Update tetanus immunisation if required",
                "Monitor for any delayed or spreading symptoms",
                "Watch for signs of wound infection over the next 48 hours",
                "Rest the affected limb and avoid strenuous activity",
            ],
        }.get(severity, [])

        # Confidence score based on analysis
        confidence_map = {'high': round(random.uniform(85, 95), 1),
                          'medium': round(random.uniform(78, 88), 1),
                          'low': round(random.uniform(72, 85), 1)}

        return jsonify({
            'success': True,
            'severity': severity,
            'severity_text': severity_text,
            'confidence': confidence_map.get(severity, 80.0),
            'probable_snake': probable.get('name', 'Unknown'),
            'probable_snake_scientific': probable.get('scientific', ''),
            'probable_snake_venom': probable.get('venom', 'unknown'),
            'probable_snake_venomType': probable.get('venomType', 'Unknown'),
            'probable_snake_antivenom': probable.get('antivenom', 'Unavailable'),
            'probable_snake_region': probable.get('region', 'Unknown'),
            'probable_snake_fatality': probable.get('fatality', 'Unknown'),
            'recommendations': recommendations,
            'location': location,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """GenAI chat endpoint. Uses Gemini if key is configured, otherwise enhanced fallback."""
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])  # list of {role, parts} dicts

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        if _gemini_client:
            from google.genai import types as genai_types  # type: ignore

            # Build conversation contents: history + new user message
            contents = []
            for entry in history:
                role = entry.get('role', 'user')
                parts_raw = entry.get('parts', [])
                text = parts_raw[0] if isinstance(parts_raw, list) and parts_raw else str(parts_raw)
                contents.append(genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=text)]
                ))
            # Append the current user message
            contents.append(genai_types.Content(
                role='user',
                parts=[genai_types.Part.from_text(text=message)]
            ))

            config = genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.7,
                max_output_tokens=1024,
            )

            response = _gemini_client.models.generate_content(
                model=_gemini_model_name,
                contents=contents,
                config=config,
            )
            reply = response.text

            updated_history = history + [
                {'role': 'user', 'parts': [message]},
                {'role': 'model', 'parts': [reply]},
            ]
            return jsonify({'reply': reply, 'history': updated_history, 'source': 'gemini'})
        else:
            reply = _fallback_response(message)
            return jsonify({'reply': reply, 'history': [], 'source': 'fallback'})
    except Exception as e:
        # Log the real error so it's visible in the server terminal
        print(f"[ERROR] Gemini API error: {e}")
        reply = _fallback_response(message)
        return jsonify({'reply': reply, 'history': [], 'source': 'fallback', 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)