import base64
import json
import mimetypes
import os
import random
import re
import urllib.error
import urllib.request

import psycopg2
import psycopg2.extras
from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from functools import wraps
from werkzeug.utils import secure_filename


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_local_env(env_path=".env"):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_local_env(os.path.join(BASE_DIR, ".env"))
load_local_env()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cle_secrete_concours_2024")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ADMIN_PASSWORD      = os.getenv("ADMIN_PASSWORD", "admin1234")

SIMULATE_AI         = os.getenv("SIMULATE_AI", "0") == "1"
USE_GEMINI          = os.getenv("USE_GEMINI", "1") == "1"
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MIN_CONFIDENCE = int(os.getenv("GEMINI_MIN_CONFIDENCE", "70"))
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest",
    ).split(",")
    if m.strip()
]

# ---------------------------------------------------------------------------
# DATABASE – PostgreSQL
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "")


def get_db():
    """Open a new PostgreSQL connection using DATABASE_URL."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL non configuree. "
            "Ajoute DATABASE_URL dans ton .env ou les variables d'environnement Railway."
        )
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    """Create tables and indexes if they don't exist yet."""
    conn = get_db()
    try:
        with conn:
            with conn.cursor() as c:
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS participants (
                        id               SERIAL PRIMARY KEY,
                        nom              TEXT NOT NULL,
                        prenom           TEXT,
                        instagram        TEXT,
                        instagram_norm   TEXT,
                        email            TEXT NOT NULL,
                        email_norm       TEXT,
                        telephone        TEXT,
                        photo_path       TEXT,
                        est_valide       INTEGER DEFAULT 0,
                        analyse_detail   TEXT,
                        date_inscription TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                # Unique indexes – ignore if already exist
                c.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_email_norm
                    ON participants (email_norm)
                    WHERE email_norm IS NOT NULL AND email_norm <> ''
                    """
                )
                c.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_instagram_norm
                    ON participants (instagram_norm)
                    WHERE instagram_norm IS NOT NULL AND instagram_norm <> ''
                    """
                )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GEMINI PROMPT
# ---------------------------------------------------------------------------

GEMINI_PROMPT = """
Tu es un verificateur d'images pour un concours.
Tu dois verifier si la photo contient TOUS les elements suivants:
1) Des gateaux (biscuits, petits gateaux, patisseries) peu importe le support (assiette, plateau, table, boite, etc.)
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

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# GEMINI
# ---------------------------------------------------------------------------

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
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
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
        if "generateContent" not in (model.get("supportedGenerationMethods") or []):
            continue
        raw_name = model.get("name", "")
        if raw_name:
            names.append(raw_name.split("/")[-1])
    return names, None


def verifier_image_ia(image_path):
    if SIMULATE_AI:
        return True, "Mode simulation actif (SIMULATE_AI=1)."
    if not USE_GEMINI:
        return False, "Verification IA desactivee (USE_GEMINI=0)."
    if not GEMINI_API_KEY:
        return False, "GEMINI_API_KEY absent. Configure la cle API Gemini."
    if GEMINI_API_KEY.startswith(("TA_CLE", "PUT_YOUR", "YOUR_")) or len(GEMINI_API_KEY) < 20:
        return False, "GEMINI_API_KEY invalide: valeur d'exemple detectee."
    if not os.path.exists(image_path):
        return False, "Image introuvable pour l'analyse."

    candidate_models = []
    for model in [GEMINI_MODEL] + GEMINI_FALLBACK_MODELS:
        if model and model not in candidate_models:
            candidate_models.append(model)

    response_json = error = status_code = selected_model = None

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
                response_json, error, status_code = _gemini_generate_content(image_path, dynamic_model)
                if not error:
                    selected_model = dynamic_model
        if error:
            if available_models:
                sample = ", ".join(available_models[:8])
                return False, f"Gemini model introuvable. Modeles detectes: {sample}"
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
    has_soltana        = _to_bool(parsed.get("has_soltana_chocolate"))
    has_maxon          = _to_bool(parsed.get("has_maxon_packet"))

    conf_plate  = max(0, min(100, _to_int(parsed.get("confidence_plate_and_cake"), 0)))
    conf_soltana = max(0, min(100, _to_int(parsed.get("confidence_soltana"), 0)))
    conf_maxon  = max(0, min(100, _to_int(parsed.get("confidence_maxon"), 0)))

    plate_ok   = has_plate_and_cake and conf_plate   >= GEMINI_MIN_CONFIDENCE
    soltana_ok = has_soltana        and conf_soltana >= GEMINI_MIN_CONFIDENCE
    maxon_ok   = has_maxon          and conf_maxon   >= GEMINI_MIN_CONFIDENCE
    valid      = _to_bool(parsed.get("valid")) and plate_ok and soltana_ok and maxon_ok

    reason = str(parsed.get("reason", "")).strip() or "Analyse Gemini effectuee."
    detail = (
        f"Gemini[{selected_model or GEMINI_MODEL}]: "
        f"assiette+gateau={'ok' if plate_ok else 'non'} ({conf_plate}%), "
        f"soltana={'ok' if soltana_ok else 'non'} ({conf_soltana}%), "
        f"maxon={'ok' if maxon_ok else 'non'} ({conf_maxon}%), "
        f"seuil={GEMINI_MIN_CONFIDENCE}%. {reason}"
    )

    if not valid:
        missing = []
        if not plate_ok:   missing.append("assiette + gateau")
        if not soltana_ok: missing.append("chocolat Soltana")
        if not maxon_ok:   missing.append("paquet Maxon")
        if missing:
            detail = f"Elements manquants: {', '.join(missing)}. {detail}"

    return valid, detail


def is_gemini_system_error(detail):
    if not detail:
        return False
    markers = (
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
    return any(m in detail.lower() for m in markers)


def has_existing_participation(email_norm, instagram_norm):
    conn = get_db()
    try:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT 1 FROM participants
                WHERE est_valide = 1
                  AND (email_norm = %s OR instagram_norm = %s)
                LIMIT 1
                """,
                (email_norm, instagram_norm),
            )
            return c.fetchone() is not None
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nom        = request.form.get("nom", "").strip()
        prenom     = request.form.get("prenom", "").strip()
        instagram  = request.form.get("instagram", "").strip()
        email      = request.form.get("email", "").strip()
        telephone  = request.form.get("telephone", "").strip()
        email_norm     = normalize_email(email)
        instagram_norm = normalize_instagram(instagram)

        required_fields = [
            (nom,       "Le nom est requis."),
            (prenom,    "Le prenom est requis."),
            (instagram, "Le compte Instagram est requis."),
            (email,     "L'email est requis."),
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
            return render_template("index.html", message="Aucune image n'a ete selectionnee.", status="error")

        file = request.files["photo"]
        if file.filename == "":
            return render_template("index.html", message="Aucune image n'a ete selectionnee.", status="error")

        if not allowed_file(file.filename):
            return render_template("index.html", message="Format non autorise. Utilisez PNG, JPG ou JPEG.", status="error")

        filename = secure_filename(file.filename)
        filename = f"{os.urandom(4).hex()}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        est_valide, analyse_detail = verifier_image_ia(filepath)

        if is_gemini_system_error(analyse_detail):
            _safe_remove(filepath)
            return render_template("index.html", message=f"Erreur configuration IA: {analyse_detail}", status="error")

        if not est_valide:
            _safe_remove(filepath)
            return render_template("index.html", message=f"Participation rejetee. {analyse_detail}", status="error")

        conn = get_db()
        try:
            with conn:
                with conn.cursor() as c:
                    c.execute(
                        """
                        INSERT INTO participants
                            (nom, prenom, instagram, instagram_norm, email, email_norm,
                             telephone, photo_path, est_valide, analyse_detail)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            nom, prenom, instagram, instagram_norm,
                            email, email_norm, telephone,
                            filename, 1 if est_valide else 0, analyse_detail,
                        ),
                    )
        except psycopg2.errors.UniqueViolation:
            conn.close()
            _safe_remove(filepath)
            return render_template(
                "index.html",
                message="Participation refusee: cet email ou ce compte Instagram a deja participe.",
                status="error",
            )
        finally:
            conn.close()

        return render_template(
            "index.html",
            message=f"Participation validee. {analyse_detail}",
            status="success",
        )

    return render_template("index.html", message=None)


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if pwd == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))
        else:
            error = "Mot de passe incorrect."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin():
    conn = get_db()
    try:
        with conn.cursor() as c:
            c.execute(
                "SELECT * FROM participants WHERE est_valide = 1 ORDER BY date_inscription DESC"
            )
            participants = c.fetchall()
    finally:
        conn.close()
    return render_template("admin.html", participants=participants, valides=participants)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe_name = os.path.basename(filename.replace("\\", "/"))
    return send_from_directory(app.config["UPLOAD_FOLDER"], safe_name)


@app.route("/tirage", methods=["POST"])
def tirage():
    conn = get_db()
    try:
        with conn.cursor() as c:
            c.execute("SELECT * FROM participants WHERE est_valide = 1")
            valides = c.fetchall()
    finally:
        conn.close()

    if not valides:
        return "Aucun participant valide pour le tirage.", 400

    gagnant   = random.choice(valides)
    prenom    = (gagnant["prenom"] or "").strip()
    nom       = (gagnant["nom"] or "").strip()
    nom_complet = f"{prenom} {nom}".strip() or nom or "Participant"
    return f"Gagnant : {nom_complet} ({gagnant['email']})"


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")