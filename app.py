import base64
import json
import mimetypes
import os
import random
import re
import sqlite3
import urllib.error
import urllib.request

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename


def load_local_env(env_path=".env"):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            # Keep already-defined environment variables as priority.
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_local_env(os.path.join(BASE_DIR, ".env"))
load_local_env()

app = Flask(__name__)
app.secret_key = "cle_secrete_concours_2024"

# CONFIGURATION
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Set SIMULATE_AI=1 to bypass IA verification
SIMULATE_AI = os.getenv("SIMULATE_AI", "0") == "1"

# Gemini config
USE_GEMINI = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MIN_CONFIDENCE = int(os.getenv("GEMINI_MIN_CONFIDENCE", "70"))
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest"
    ).split(",")
    if model.strip()
]


GEMINI_PROMPT = """
Tu es un verificateur d'images pour un concours.
Tu dois verifier si la photo contient TOUS les elements suivants:
1) Une assiette avec un gateau
2) Le chocolat Soltana
3) Le paquet de gateau Maxon

Reponds uniquement en JSON strict, sans markdown, avec ce schema exact:
{
  "has_plate_and_cake": true,
  "has_soltana_chocolate": true,
  "has_maxon_packet": true,
  "confidence_plate_and_cake": 92,
  "confidence_soltana": 95,
  "confidence_maxon": 91,
  "valid": true,
  "reason": "explication courte"
}

Contraintes:
- confidence_* doit etre un entier entre 0 et 100.
- Sois strict: true seulement si l'element est clairement visible.
- valid=true uniquement si les 3 premiers champs sont true.
""".strip()


def init_db():
    conn = sqlite3.connect("concours.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            prenom TEXT,
            instagram TEXT,
            instagram_norm TEXT,
            email TEXT NOT NULL,
            email_norm TEXT,
            telephone TEXT,
            photo_path TEXT,
            est_valide INTEGER DEFAULT 0,
            analyse_detail TEXT,
            date_inscription TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Backward compatibility for existing DB
    c.execute("PRAGMA table_info(participants)")
    existing_columns = {row[1] for row in c.fetchall()}
    for column_name in (
        "prenom",
        "instagram",
        "instagram_norm",
        "email_norm",
        "telephone",
        "analyse_detail",
    ):
        if column_name not in existing_columns:
            c.execute(f"ALTER TABLE participants ADD COLUMN {column_name} TEXT")

    # Fill normalized columns for old rows if needed.
    c.execute(
        """
        UPDATE participants
        SET email_norm = LOWER(TRIM(email))
        WHERE (email_norm IS NULL OR email_norm = '') AND email IS NOT NULL
    """
    )
    c.execute(
        """
        UPDATE participants
        SET instagram_norm = LOWER(TRIM(REPLACE(instagram, '@', '')))
        WHERE (instagram_norm IS NULL OR instagram_norm = '') AND instagram IS NOT NULL
    """
    )

    # Try DB-level uniqueness. If old duplicates exist, keep app-level checks.
    try:
        c.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_unique_email_norm
            ON participants(email_norm)
            WHERE email_norm IS NOT NULL AND email_norm <> ''
        """
        )
        c.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_unique_instagram_norm
            ON participants(instagram_norm)
            WHERE instagram_norm IS NOT NULL AND instagram_norm <> ''
        """
        )
    except sqlite3.IntegrityError:
        pass

    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_email(value):
    return value.strip().lower()


def normalize_instagram(value):
    return value.strip().lower().lstrip("@")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "oui"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_json_obj(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _gemini_generate_content(image_path, model_name):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": GEMINI_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body), None, None
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return None, f"Gemini HTTP {exc.code}: {error_body[:300]}", exc.code
    except urllib.error.URLError as exc:
        return None, f"Gemini network error: {exc}", None
    except Exception as exc:
        return None, f"Gemini unexpected error: {exc}", None


def _gemini_list_models():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
    except Exception as exc:
        return [], f"Gemini list models error: {exc}"

    names = []
    for model in payload.get("models", []):
        methods = model.get("supportedGenerationMethods", []) or []
        if "generateContent" not in methods:
            continue

        raw_name = model.get("name", "")
        if not raw_name:
            continue
        names.append(raw_name.split("/")[-1])

    return names, None


def verifier_image_ia(image_path):
    if SIMULATE_AI:
        return True, "Mode simulation actif (SIMULATE_AI=1)."

    if not USE_GEMINI:
        return False, "Verification IA desactivee (USE_GEMINI=0)."

    if not GEMINI_API_KEY:
        return (
            False,
            "GEMINI_API_KEY absent. Configure la cle API Gemini dans les variables d'environnement.",
        )

    if GEMINI_API_KEY.startswith(("TA_CLE", "PUT_YOUR", "YOUR_")) or len(GEMINI_API_KEY) < 20:
        return (
            False,
            "GEMINI_API_KEY invalide: valeur d'exemple detectee. Remplace-la par une vraie cle Google AI Studio.",
        )

    if not os.path.exists(image_path):
        return False, "Image introuvable pour l'analyse."

    candidate_models = []
    for model in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        if model and model not in candidate_models:
            candidate_models.append(model)

    response_json = None
    error = None
    status_code = None
    selected_model = None

    for model_name in candidate_models:
        response_json, error, status_code = _gemini_generate_content(image_path, model_name)
        if not error:
            selected_model = model_name
            break
        if status_code != 404:
            return False, error

    if error and status_code == 404:
        available_models, list_error = _gemini_list_models()
        if available_models:
            flash_candidates = [m for m in available_models if "flash" in m.lower()]
            dynamic_model = flash_candidates[0] if flash_candidates else available_models[0]

            if dynamic_model not in candidate_models:
                response_json, error, status_code = _gemini_generate_content(
                    image_path, dynamic_model
                )
                if not error:
                    selected_model = dynamic_model

        if error:
            if available_models:
                sample = ", ".join(available_models[:8])
                return (
                    False,
                    "Gemini model introuvable pour generateContent. "
                    f"Modele configure: {GEMINI_MODEL}. Modeles detectes: {sample}",
                )
            return False, f"{error}. {list_error or 'Impossible de lister les modeles.'}"

    if error:
        return False, error

    if not response_json:
        return False, "Reponse Gemini vide."

    text_chunks = []
    for candidate in response_json.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text_chunks.append(part["text"])

    if not text_chunks:
        return False, "Gemini: aucune sortie textuelle exploitable."

    parsed = _extract_json_obj("\n".join(text_chunks))
    if not isinstance(parsed, dict):
        return False, "Gemini: JSON non lisible."

    has_plate_and_cake = _to_bool(parsed.get("has_plate_and_cake"))
    has_soltana = _to_bool(parsed.get("has_soltana_chocolate"))
    has_maxon = _to_bool(parsed.get("has_maxon_packet"))

    conf_plate = max(0, min(100, _to_int(parsed.get("confidence_plate_and_cake"), 0)))
    conf_soltana = max(0, min(100, _to_int(parsed.get("confidence_soltana"), 0)))
    conf_maxon = max(0, min(100, _to_int(parsed.get("confidence_maxon"), 0)))

    plate_ok = has_plate_and_cake and conf_plate >= GEMINI_MIN_CONFIDENCE
    soltana_ok = has_soltana and conf_soltana >= GEMINI_MIN_CONFIDENCE
    maxon_ok = has_maxon and conf_maxon >= GEMINI_MIN_CONFIDENCE
    valid_from_fields = plate_ok and soltana_ok and maxon_ok

    # The final rule is always enforced server-side
    valid = _to_bool(parsed.get("valid")) and valid_from_fields

    reason = str(parsed.get("reason", "")).strip() or "Analyse Gemini effectuee."
    detail = (
        f"Gemini[{selected_model or GEMINI_MODEL}]: assiette+gateau={'ok' if plate_ok else 'non'} ({conf_plate}%), "
        f"soltana={'ok' if soltana_ok else 'non'} ({conf_soltana}%), "
        f"maxon={'ok' if maxon_ok else 'non'} ({conf_maxon}%), "
        f"seuil={GEMINI_MIN_CONFIDENCE}%. {reason}"
    )

    if not valid:
        missing = []
        if not plate_ok:
            missing.append("assiette + gateau")
        if not soltana_ok:
            missing.append("chocolat Soltana")
        if not maxon_ok:
            missing.append("paquet Maxon")
        if missing:
            detail = f"Elements manquants: {', '.join(missing)}. {detail}"

    return valid, detail


def is_gemini_system_error(detail):
    if not detail:
        return False
    detail_lower = detail.lower()
    system_markers = (
        "gemini_api_key absent",
        "verification ia desactivee",
        "gemini http",
        "gemini model introuvable",
        "gemini network error",
        "gemini unexpected error",
        "reponse gemini vide",
        "gemini: json non lisible",
        "gemini: aucune sortie textuelle exploitable",
    )
    return any(marker in detail_lower for marker in system_markers)


def has_existing_participation(email_norm, instagram_norm):
    conn = sqlite3.connect("concours.db")
    c = conn.cursor()
    c.execute(
        """
        SELECT 1
        FROM participants
        WHERE est_valide = 1
          AND (email_norm = ? OR instagram_norm = ?)
        LIMIT 1
    """,
        (email_norm, instagram_norm),
    )
    exists = c.fetchone() is not None
    conn.close()
    return exists


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nom = request.form.get("nom", "").strip()
        prenom = request.form.get("prenom", "").strip()
        instagram = request.form.get("instagram", "").strip()
        email = request.form.get("email", "").strip()
        telephone = request.form.get("telephone", "").strip()
        email_norm = normalize_email(email)
        instagram_norm = normalize_instagram(instagram)

        required_fields = [
            (nom, "Le nom est requis."),
            (prenom, "Le prenom est requis."),
            (instagram, "Le compte Instagram est requis."),
            (email, "L'email est requis."),
            (telephone, "Le numero de telephone est requis."),
        ]
        for value, error_message in required_fields:
            if not value:
                return render_template("index.html", message=error_message, status="error")

        if has_existing_participation(email_norm, instagram_norm):
            return render_template(
                "index.html",
                message="Participation refusee: cet email ou ce compte Instagram a deja participe.",
                status="error",
            )

        if "photo" not in request.files:
            return render_template(
                "index.html",
                message="Aucune image n'a ete selectionnee.",
                status="error",
            )

        file = request.files["photo"]
        if file.filename == "":
            return render_template(
                "index.html",
                message="Aucune image n'a ete selectionnee.",
                status="error",
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                message="Format non autorise. Utilisez PNG, JPG ou JPEG.",
                status="error",
            )

        filename = secure_filename(file.filename)
        filename = f"{os.urandom(4).hex()}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        est_valide, analyse_detail = verifier_image_ia(filepath)

        if is_gemini_system_error(analyse_detail):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass

            return render_template(
                "index.html",
                message=f"Erreur configuration IA: {analyse_detail}",
                status="error",
            )

        if not est_valide:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass

            return render_template(
                "index.html",
                message=f"Participation rejetee. {analyse_detail}",
                status="error",
            )

        conn = sqlite3.connect("concours.db")
        c = conn.cursor()
        try:
            c.execute(
                """
                INSERT INTO participants (
                    nom, prenom, instagram, instagram_norm, email, email_norm,
                    telephone, photo_path, est_valide, analyse_detail
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    nom,
                    prenom,
                    instagram,
                    instagram_norm,
                    email,
                    email_norm,
                    telephone,
                    filename,
                    1 if est_valide else 0,
                    analyse_detail,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass
            return render_template(
                "index.html",
                message="Participation refusee: cet email ou ce compte Instagram a deja participe.",
                status="error",
            )

        conn.close()

        message = f"Participation validee. {analyse_detail}"
        return render_template("index.html", message=message, status="success")

    return render_template("index.html", message=None)


@app.route("/admin")
def admin():
    conn = sqlite3.connect("concours.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM participants WHERE est_valide = 1 ORDER BY date_inscription DESC")
    participants = c.fetchall()
    valides = participants
    conn.close()
    return render_template("admin.html", participants=participants, valides=valides)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # Compatibility: accepts older stored paths
    safe_name = os.path.basename(filename.replace("\\", "/"))
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


@app.route("/tirage", methods=["POST"])
def tirage():
    conn = sqlite3.connect("concours.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM participants WHERE est_valide = 1")
    valides = c.fetchall()

    if not valides:
        conn.close()
        return "Aucun participant valide pour le tirage.", 400

    gagnant = random.choice(valides)
    conn.close()

    prenom = (gagnant["prenom"] or "").strip()
    nom = (gagnant["nom"] or "").strip()
    nom_complet = f"{prenom} {nom}".strip() or nom or "Participant"
    return f"Gagnant : {nom_complet} ({gagnant['email']})"


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    init_db()
    app.run(debug=True)
