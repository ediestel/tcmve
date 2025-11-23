# main.py — FINAL VERSION FOR YOUR EXACT TABLE
import os
import json
import html
import asyncio
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from backend.tcmve import TCMVE
from asyncio import Queue

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = TCMVE(max_rounds=10)
active_ws: WebSocket | None = None

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "tcmve"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

# WebSocket — live streaming
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    global active_ws
    await websocket.accept()
    active_ws = websocket
    engine.active_ws = websocket
    try:
        while True:
            # read incoming messages so browser won't drop the connection
            try:
                await websocket.receive_text()
            except:
                await asyncio.sleep(1)
    finally:
        active_ws = None
        engine.active_ws = None

# === LIST OF DB COLUMNS (NO ALTERATION) ===
DB_COLUMNS = {
    "usegenerator",
    "useverifier",
    "usearbiter",
    "generatorprovider",
    "verifierprovider",
    "arbiterprovider",
    "maxrounds",
    "maritalfreedom",
    "vicecheck",
    "selfrefine",
    "streammode"
}

# SAVE config — matches your exact table
# SAVE config — matches your exact table 100%
def save_config(flags: dict):
    # Normalize camelCase → snake_case exactly as your DB expects
    normalized = {}
    key_map = {
        "useGenerator": "usegenerator",
        "useVerifier": "useverifier",
        "useArbiter": "usearbiter",
        "generatorProvider": "generatorprovider",
        "verifierProvider": "verifierprovider",
        "arbiterProvider": "arbiterprovider",
        "maxRounds": "maxrounds",
        "maritalFreedom": "maritalfreedom",
        "viceCheck": "vicecheck",
        "selfRefine": "selfrefine",
        "streamMode": "streammode",
        "gameMode": "gamemode",
        "selectedGame": "selectedgame",
        "eiqLevel": "eiqlevel",
        "simulatedPersons": "simulatedpersons",
        "meanBiq": "meanbiq",
        "sigmaBiq": "sigmabiq",
        "tlpoFull": "tlpofull",
        "noXml": "noxml",
        "sevenDomains": "sevendomains",
        "virtuesIndependent": "virtuesindependent",
        "biqDistribution": "biqdistribution",
        "output": "output",
        "nashMode": "nashmode",
    }
    # STRICT normalization — only mapped keys allowed
    for k, v in flags.items():
        if k not in key_map:
            # ignore unknown UI keys so SQL does not break
            continue
        db_key = key_map[k]
        normalized[db_key] = v

    # Defaults (use *snake_case* keys!)
    defaults = {
        "usegenerator": True,
        "useverifier": True,
        "usearbiter": True,
        "generatorprovider": "openai",
        "verifierprovider": "xai",
        "arbiterprovider": "xai",

        "maxrounds": 5,
        "maritalfreedom": False,
        "vicecheck": True,
        "selfrefine": True,

        "streammode": "arbiter_only",
        "gamemode": "dynamic",
        "selectedgame": None,

        "eiqlevel": 10,
        "simulatedpersons": 50,  # Increased for research flexibility
        "meanbiq": 100,
        "sigmabiq": 15,
        "tlpofull": False,
        "noxml": False,
        "sevendomains": True,
        "virtuesindependent": True,
        "biqdistribution": "gaussian",
        "output": "result",
        "nashmode": "auto",
    }

    # Merge
    normalized = {**defaults, **normalized}

    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute(f"""
                INSERT INTO configs (id, {', '.join(normalized.keys())})
                VALUES (1, {', '.join(['%(' + k + ')s' for k in normalized.keys()])})
                ON CONFLICT (id) DO UPDATE SET
                {', '.join([k + ' = EXCLUDED.' + k for k in normalized.keys()])}
            """, normalized)
            conn.commit()

# LOAD config
@app.get("/config")
def get_config():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT * FROM configs WHERE id = 1")
            row = c.fetchone()
            if row:
                d = dict(row)
                d.pop("id", None)
                d.pop("timestamp", None)
                return d
            return {}

# SAVE config
@app.post("/config")
async def save_config_endpoint(request: Request):
    data = await request.json()
    save_config(data)
    return {"status": "Config saved"}





# UPDATE STATS AFTER EVERY RUN
def update_dashboard_stats():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:

            c.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    COALESCE(AVG(eiq), 0) as avg_eiq,
                    COALESCE(AVG(tqi), 0) as avg_tlpo,
                    COALESCE(SUM(tokens_used), 0) as total_tokens,
                    COALESCE(SUM(cost_estimate), 0) as total_cost
                FROM runs
            """)
            stats = c.fetchone()

            c.execute("""
                SELECT json_agg(t) as recent_runs
                FROM (
                    SELECT id, query, final_answer, eiq, tqi, tokens_used, cost_estimate
                    FROM runs 
                    ORDER BY id DESC 
                    LIMIT 5
                ) t
            """)
            recent = c.fetchone()

            c.execute("""
                INSERT INTO dashboard_stats (
                    id, total_runs, avg_eiq, avg_tlpo, total_tokens, total_cost, recent_runs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    total_runs = EXCLUDED.total_runs,
                    avg_eiq = EXCLUDED.avg_eiq,
                    avg_tlpo = EXCLUDED.avg_tlpo,
                    total_tokens = EXCLUDED.total_tokens,
                    total_cost = EXCLUDED.total_cost,
                    recent_runs = EXCLUDED.recent_runs,
                    updated_at = NOW()
            """, (
                1,
                stats["total_runs"],
                round(float(stats["avg_eiq"] or 0)),
                round(float(stats["avg_tlpo"] or 0), 4),
                int(stats["total_tokens"] or 0),
                round(float(stats["total_cost"] or 0), 4),
                json.dumps(recent["recent_runs"] or [])
            ))

            conn.commit()



# MAIN RUN — saves config + runs engine
@app.post("/run")
async def run_engine(request: Request):
    global engine

    data = await request.json()
    query = html.escape(data.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "Query required")

    user_id = data.get("user_id")
    session_id = data.get("session_id")

    virtues = data.get("virtues", {})
    flags = data.get("flags", {})

    # Load existing config from DB as base
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT * FROM configs WHERE id = 1")
            row = c.fetchone()
            if row:
                db_flags = dict(row)
                db_flags.pop("id", None)
                db_flags.pop("timestamp", None)
                # Reverse normalize snake_case → camelCase for merging
                key_map = {
                    "useGenerator": "usegenerator",
                    "useVerifier": "useverifier",
                    "useArbiter": "usearbiter",
                    "generatorProvider": "generatorprovider",
                    "verifierProvider": "verifierprovider",
                    "arbiterProvider": "arbiterprovider",
                    "maxRounds": "maxrounds",
                    "maritalFreedom": "maritalfreedom",
                    "viceCheck": "vicecheck",
                    "selfRefine": "selfrefine",
                    "streamMode": "streammode",
                    "gameMode": "gamemode",
                    "selectedGame": "selectedgame",
                    "eiqLevel": "eiqlevel",
                    "simulatedPersons": "simulatedpersons",
                    "meanBiq": "meanbiq",
                    "sigmaBiq": "sigmabiq",
                    "tlpoFull": "tlpofull",
                    "noXml": "noxml",
                    "sevenDomains": "sevendomains",
                    "virtuesIndependent": "virtuesindependent",
                    "biqDistribution": "biqdistribution",
                    "output": "output",
                    "nashMode": "nashmode",
                }
                reverse_map = {v: k for k, v in key_map.items()}
                camel_flags = {reverse_map.get(k, k): v for k, v in db_flags.items() if k in reverse_map}
                # Merge: request flags override DB
                flags = {**camel_flags, **flags}
            else:
                # Use defaults if no DB config
                defaults = {
                    "useGenerator": True,
                    "useVerifier": True,
                    "useArbiter": True,
                    "generatorProvider": "openai",
                    "verifierProvider": "xai",
                    "arbiterProvider": "xai",
                    "maxRounds": 5,
                    "maritalFreedom": False,
                    "viceCheck": True,
                    "selfRefine": True,
                    "streamMode": "arbiter_only",
                }
                flags = {**defaults, **flags}

    # Save config INTO DB — missing becomes NULL
    save_config(flags)

    # Build args object — attach ALL UI flags (no renaming)
    class Args:
        pass

    args = Args()
    for k, v in flags.items():
        setattr(args, k.lower(), v)

    # Set arbiter-only mode attributes
    if getattr(args, "arbiter_only", False):
        args.use_generator = False
        args.use_verifier = False
        args.use_arbiter = True
    else:
        args.use_generator = True
        args.use_verifier = True
        args.use_arbiter = True

    # === STREAMING ===
    stream_mode = getattr(args, "streammode", "arbiter_only")
    current_role = None
    chunk_index = 0

    if stream_mode != "none":
        token_queue = Queue()

        async def send_token(token: str):
            nonlocal chunk_index
            if stream_mode == "arbiter_only" and current_role != "arbiter":
                return
            prefix = {"generator": "[Generator] ", "verifier": "[Verifier] ", "arbiter": "[Arbiter] "}.get(current_role, "")
            full_token = f"{prefix}{token}"
            await token_queue.put(full_token)

            # Send via WebSocket if connected
            if active_ws:
                await active_ws.send_text(f"TOKEN:{prefix}{token}")

            # Always save to database
            try:
                with get_conn() as conn:
                    with conn.cursor() as c:
                        c.execute("""
                            INSERT INTO streaming_responses (session_id, agent_role, response_chunk, chunk_index)
                            VALUES (%s, %s, %s, %s)
                        """, (session_id or "anonymous", current_role, f"TOKEN:{prefix}{token}", chunk_index))
                        conn.commit()
                chunk_index += 1
            except Exception as e:
                print(f"Error saving to DB: {e}")

        async def token_generator():
            try:
                while True:
                    token = await token_queue.get()
                    if token == "END":
                        break
                    yield token
            except Exception as e:
                print(f"Generator error: {e}")

        # Wrap invoke to track speaker
        def wrap_role(role, orig):
            async def wrapper(*a, **k):
                nonlocal current_role
                current_role = role
                r = await orig(*a, **k)
                current_role = None
                return r
            return wrapper

            # Patch invoke wrappers only if object exists
        if hasattr(engine.generator, "invoke"):
                engine.generator.invoke = wrap_role("generator", engine.generator.invoke)
        if hasattr(engine.verifier, "invoke"):
                engine.verifier.invoke = wrap_role("verifier", engine.verifier.invoke)
        if hasattr(engine.arbiter, "invoke"):
                engine.arbiter.invoke = wrap_role("arbiter", engine.arbiter.invoke)

        engine.token_callback = send_token

    # Set user and session context
    if user_id:
        engine.user_id = user_id
    if session_id:
        engine.session_id = session_id

    # Inject virtues
    def to_dict(a):
        return {
            "P": round(a[0], 2),
            "J": round(a[1], 2),
            "F": round(a[2], 2),
            "T": round(a[3], 2),
            "V": round(a[4], 2),
            "L": round(a[5], 2),
            "H": round(a[6], 2),
            "Ω": round(a[7], 2),
        }

    for r in ["generator", "verifier", "arbiter"]:
        if r in virtues:
            engine.virtue_vectors[r] = to_dict(virtues[r])

    if stream_mode != "none":
        # Run in task
        async def run_engine_task():
            try:
                result = engine.run(query, args=args)

                # Save run
                with get_conn() as conn:
                    with conn.cursor() as c:
                        c.execute(
                            """
                            INSERT INTO runs (
                                query, description,
                                generator_provider, verifier_provider, arbiter_provider, maxrounds,
                                virtues_generator, virtues_verifier, virtues_arbiter,
                                final_answer, converged, rounds,
                                tlpo_scores, tlpo_markup,
                                eiq, tqi, tcs, fd, es,
                                tokens_used, cost_estimate
                            )
                            VALUES (
                                %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s, %s, %s, %s,
                                %s, %s
                            )
                            """,
                            (
                                query,
                                result.get("description")
                                or f"Run — {getattr(args, 'arbiterprovider', 'unknown')} — eIQ {result.get('eIQ', '?')}",
                                getattr(args, "generatorprovider", None),
                                getattr(args, "verifierprovider", None),
                                getattr(args, "arbiterprovider", None),
                                getattr(args, "maxrounds", None),
                                json.dumps(virtues.get("generator")),
                                json.dumps(virtues.get("verifier")),
                                json.dumps(virtues.get("arbiter")),
                                result.get("final_answer"),
                                result.get("converged", False),
                                result.get("rounds", 0),
                                json.dumps(result.get("tlpo_scores", {})),
                                result.get("tlpo_markup", ""),
                                result.get("eIQ"),
                                result.get("TQI"),
                                result.get("metrics", {}).get("TCS"),
                                result.get("metrics", {}).get("FD"),
                                result.get("metrics", {}).get("ES"),
                                result.get("tokens_used"),
                                result.get("cost_estimate"),
                            ),
                        )
                        conn.commit()

                update_dashboard_stats()

                await token_queue.put("END")

            except Exception as e:
                print(f"Run error: {e}")
                await token_queue.put("END")

        asyncio.create_task(run_engine_task())

        return StreamingResponse(token_generator(), media_type="text/plain")

    else:
        result = engine.run(query, args=args)

        # Save run
        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute(
                    """
                    INSERT INTO runs (
                        query, description,
                        generator_provider, verifier_provider, arbiter_provider, maxrounds,
                        virtues_generator, virtues_verifier, virtues_arbiter,
                        final_answer, converged, rounds,
                        tlpo_scores, tlpo_markup,
                        eiq, tqi, tcs, fd, es,
                        tokens_used, cost_estimate
                    )
                    VALUES (
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (
                        query,
                        result.get("description")
                        or f"Run — {getattr(args, 'arbiterprovider', 'unknown')} — eIQ {result.get('eIQ', '?')}",
                        getattr(args, "generatorprovider", None),
                        getattr(args, "verifierprovider", None),
                        getattr(args, "arbiterprovider", None),
                        getattr(args, "maxrounds", None),
                        json.dumps(virtues.get("generator")),
                        json.dumps(virtues.get("verifier")),
                        json.dumps(virtues.get("arbiter")),
                        result.get("final_answer"),
                        result.get("converged", False),
                        result.get("rounds", 0),
                        json.dumps(result.get("tlpo_scores", {})),
                        result.get("tlpo_markup", ""),
                        result.get("eIQ"),
                        result.get("TQI"),
                        result.get("metrics", {}).get("TCS"),
                        result.get("metrics", {}).get("FD"),
                        result.get("metrics", {}).get("ES"),
                        result.get("tokens_used"),
                        result.get("cost_estimate"),
                    ),
                )
                conn.commit()

        update_dashboard_stats()

        return result


# GET DASHBOARD STATS — FAST, CACHED, NO RECALCULATION
@app.get("/runs")
def get_runs(limit: int = None):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            query = """
                SELECT id, query, description, final_answer, eiq, tqi, tokens_used, cost_estimate,
                       tlpo_scores, tlpo_markup, created_at as timestamp
                FROM runs
                ORDER BY id DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            c.execute(query)
            rows = c.fetchall()

            # Transform to match frontend expectations
            runs = []
            for row in rows:
                run = dict(row)
                # Parse JSON fields
                tlpo_scores = json.loads(run.get("tlpo_scores", "{}")) if run.get("tlpo_scores") else {}
                # Create nested result structure like dashboard stats
                run["result"] = {
                    "final_answer": run.get("final_answer", ""),
                    "eIQ": run.get("eiq", 0),
                    "TLPO": run.get("tqi", 0),  # Assuming TLPO is TQI
                    "tokens_used": run.get("tokens_used", 0),
                    "cost_estimate": run.get("cost_estimate", 0),
                    "four_causes": tlpo_scores.get("four_causes", {}),
                    "ontology": tlpo_scores.get("ontology", []),
                    "flags": tlpo_scores.get("flags", [])
                }
                # Remove fields that are now in result
                run.pop("final_answer", None)
                run.pop("eiq", None)
                run.pop("tqi", None)
                run.pop("tokens_used", None)
                run.pop("cost_estimate", None)
                run.pop("tlpo_scores", None)
                run.pop("tlpo_markup", None)
                runs.append(run)

            return runs

@app.delete("/runs/{run_id}")
def delete_run(run_id: int):
    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("DELETE FROM runs WHERE id = %s", (run_id,))
            conn.commit()
    return {"status": "Run deleted"}

@app.get("/dashboard/stats")
def get_dashboard_stats():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT * FROM dashboard_stats WHERE id = 1")
            row = c.fetchone()

            if row:
                data = dict(row)
                data.pop("id", None)
                data["recent_runs"] = json.loads(data.get("recent_runs", "[]"))
                return data

            return {
                "total_runs": 0,
                "avg_eiq": 0,
                "avg_tlpo": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "recent_runs": [],
            }


@app.get("/presets")
def get_presets():
    """Get available virtue presets for domain-specific analysis."""
    from backend.virtue_presets import list_presets
    return {"presets": list_presets()}

@app.get("/presets/{preset_name}")
def get_preset(preset_name: str):
    """Get a specific virtue preset configuration."""
    from backend.virtue_presets import get_preset
    try:
        return get_preset(preset_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/apply-preset/{preset_name}")
def apply_preset(preset_name: str):
    """Apply a virtue preset to the current TCMVE engine."""
    try:
        # Use the engine's apply_virtue_preset method to ensure current_preset is set
        engine.apply_virtue_preset(preset_name)
        return {"status": f"Preset '{preset_name}' applied successfully", "virtue_vectors": engine.virtue_vectors}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Recommended Sets Endpoints
@app.get("/recommended-sets")
def get_recommended_sets():
    """Get available recommended game sets."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT id, name, description, games, use_case FROM recommended_sets ORDER BY name")
            rows = c.fetchall()
            return {"sets": [dict(row) for row in rows]}

@app.get("/recommended-sets/{set_id}")
def get_recommended_set(set_id: int):
    """Get a specific recommended set by ID."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT id, name, description, games, use_case FROM recommended_sets WHERE id = %s", (set_id,))
            row = c.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Recommended set not found")
            return dict(row)

@app.post("/recommended-sets")
async def create_recommended_set(request: Request):
    """Create a new recommended set."""
    data = await request.json()
    name = data.get("name")
    description = data.get("description")
    games = data.get("games", [])
    use_case = data.get("use_case")

    if not name or not games:
        raise HTTPException(status_code=400, detail="Name and games are required")

    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO recommended_sets (name, description, games, use_case)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (name, description, json.dumps(games), use_case))
            set_id = c.fetchone()[0]
            conn.commit()
    return {"id": set_id, "message": "Recommended set created"}

@app.put("/recommended-sets/{set_id}")
async def update_recommended_set(set_id: int, request: Request):
    """Update a recommended set."""
    data = await request.json()
    name = data.get("name")
    description = data.get("description")
    games = data.get("games", [])
    use_case = data.get("use_case")

    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("""
                UPDATE recommended_sets
                SET name = %s, description = %s, games = %s, use_case = %s, updated_at = NOW()
                WHERE id = %s
            """, (name, description, json.dumps(games), use_case, set_id))
            if c.rowcount == 0:
                raise HTTPException(status_code=404, detail="Recommended set not found")
            conn.commit()
    return {"message": "Recommended set updated"}

@app.delete("/recommended-sets/{set_id}")
def delete_recommended_set(set_id: int):
    """Delete a recommended set."""
    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("DELETE FROM recommended_sets WHERE id = %s", (set_id,))
            if c.rowcount == 0:
                raise HTTPException(status_code=404, detail="Recommended set not found")
            conn.commit()
    return {"message": "Recommended set deleted"}

@app.get("/trials")
def get_trials():
    """Get list of ARCHER trial results."""
    import glob
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    trial_files = glob.glob(str(results_dir / "*archer*.json"))
    trials = []
    for file_path in trial_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data['filename'] = Path(file_path).name
                trials.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return {"trials": trials}

@app.get("/trials/{filename}")
def get_trial(filename: str):
    """Get specific trial result."""
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    file_path = results_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Trial not found")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trial: {e}")

# DEFAULTS ENDPOINTS
@app.get("/defaults")
def get_defaults():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT * FROM defaults WHERE id = 1")
            row = c.fetchone()
            if row:
                d = dict(row)
                d.pop("id", None)
                d.pop("timestamp", None)
                return d
            return {
                "generator": {"Ω": 0.97, "P": 0.8, "J": 0.75, "F": 0.65, "T": 0.85, "L": 0.72, "V": 0.85, "H": 0.89},
                "verifier": {"Ω": 0.95, "P": 0.9, "J": 0.95, "F": 0.8, "T": 0.9, "L": 0.65, "V": 0.9, "H": 0.95},
                "arbiter": {"Ω": 0.95, "P": 0.85, "J": 0.8, "F": 0.9, "T": 0.85, "L": 0.85, "V": 0.8, "H": 0.85}
            }

@app.post("/defaults")
async def save_defaults(request: Request):
    data = await request.json()
    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO defaults (id, generator, verifier, arbiter)
                VALUES (1, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    generator = EXCLUDED.generator,
                    verifier = EXCLUDED.verifier,
                    arbiter = EXCLUDED.arbiter
            """, (
                json.dumps(data.get("generator", {})),
                json.dumps(data.get("verifier", {})),
                json.dumps(data.get("arbiter", {}))
            ))
            conn.commit()
    return {"status": "Defaults saved"}

@app.delete("/defaults/{role}")
def delete_default(role: str):
    if role not in ["generator", "verifier", "arbiter"]:
        raise HTTPException(400, "Invalid role")
    with get_conn() as conn:
        with conn.cursor() as c:
            # Reset to default values
            defaults = {
                "generator": {"Ω": 0.97, "P": 0.8, "J": 0.75, "F": 0.65, "T": 0.85, "L": 0.72, "V": 0.85, "H": 0.89},
                "verifier": {"Ω": 0.95, "P": 0.9, "J": 0.95, "F": 0.8, "T": 0.9, "L": 0.65, "V": 0.9, "H": 0.95},
                "arbiter": {"Ω": 0.95, "P": 0.85, "J": 0.8, "F": 0.9, "T": 0.85, "L": 0.85, "V": 0.8, "H": 0.85}
            }
            current = get_defaults()
            current[role] = defaults[role]
            c.execute("""
                INSERT INTO defaults (id, generator, verifier, arbiter)
                VALUES (1, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    generator = EXCLUDED.generator,
                    verifier = EXCLUDED.verifier,
                    arbiter = EXCLUDED.arbiter
            """, (
                json.dumps(current.get("generator", {})),
                json.dumps(current.get("verifier", {})),
                json.dumps(current.get("arbiter", {}))
            ))
            conn.commit()
    return {"status": f"Default for {role} reset"}

# Reasoning Excellence Endpoints
from backend.virtue_evolution import virtue_tracker

@app.get("/reasoning-excellence/{session_id}")
def get_reasoning_excellence(session_id: str, agent_role: str = None, virtue_name: str = None, limit: int = 100):
    """Get reasoning excellence evolution history for a session"""
    try:
        evolution = virtue_tracker.get_virtue_evolution(session_id, agent_role, virtue_name, limit)
        return {"evolution": evolution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve reasoning excellence: {str(e)}")

@app.get("/reasoning-excellence/{session_id}/current-state")
def get_current_reasoning_state(session_id: str):
    """Get current reasoning excellence state for all agents in a session"""
    try:
        current_state = virtue_tracker.get_current_virtue_state(session_id)
        return {"current_state": current_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve current reasoning state: {str(e)}")

@app.get("/reasoning-excellence/{session_id}/analysis")
def analyze_reasoning_development(session_id: str, agent_role: str = None):
    """Analyze reasoning excellence development patterns for a session"""
    try:
        analysis = virtue_tracker.analyze_virtue_development(session_id, agent_role)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze reasoning development: {str(e)}")

# Virtue Evolution Endpoints (alias for reasoning-excellence)
@app.get("/virtue-evolution/{session_id}")
def get_virtue_evolution(session_id: str, agent_role: str = None, virtue_name: str = None, limit: int = 100):
    """Get virtue evolution history for a session"""
    return get_reasoning_excellence(session_id, agent_role, virtue_name, limit)

@app.get("/virtue-evolution/{session_id}/current-state")
def get_virtue_current_state(session_id: str):
    """Get current virtue state for all agents in a session"""
    return get_current_reasoning_state(session_id)

@app.get("/virtue-evolution/{session_id}/analysis")
def analyze_virtue_development(session_id: str, agent_role: str = None):
    """Analyze virtue development patterns for a session"""
    return analyze_reasoning_development(session_id, agent_role)

# User-based Virtue Evolution Endpoints
@app.get("/users/{user_id}/virtue-evolution")
def get_user_virtue_evolution(user_id: int, agent_role: str = None, virtue_name: str = None, limit: int = 100):
    """Get virtue evolution history for all sessions of a user"""
    try:
        # Get all sessions for this user
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as c:
                c.execute("SELECT session_id FROM user_sessions WHERE user_id = %s", (user_id,))
                sessions = c.fetchall()

        if not sessions:
            return {"evolution": []}

        # Aggregate evolution from all user sessions
        all_evolution = []
        for session in sessions:
            try:
                evolution = virtue_tracker.get_virtue_evolution(session['session_id'], agent_role, virtue_name, limit)
                all_evolution.extend(evolution)
            except Exception as e:
                continue  # Skip sessions with errors

        # Sort by timestamp and limit
        all_evolution.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return {"evolution": all_evolution[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user virtue evolution: {str(e)}")

@app.get("/users/{user_id}/virtue-evolution/current-state")
def get_user_current_virtue_state(user_id: int):
    """Get current virtue state across all user sessions"""
    try:
        # Get the most recent session for this user
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as c:
                c.execute("""
                    SELECT session_id FROM user_sessions
                    WHERE user_id = %s
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (user_id,))
                session = c.fetchone()

        if not session:
            return {"current_state": {}}

        return get_current_reasoning_state(session['session_id'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user current virtue state: {str(e)}")

@app.get("/users/{user_id}/virtue-evolution/analysis")
def analyze_user_virtue_development(user_id: int, agent_role: str = None):
    """Analyze virtue development patterns across all user sessions"""
    try:
        # Get all sessions for this user
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as c:
                c.execute("SELECT session_id FROM user_sessions WHERE user_id = %s", (user_id,))
                sessions = c.fetchall()

        if not sessions:
            return {"analysis": {}}

        # Analyze the most recent session (or aggregate if needed)
        recent_session = max(sessions, key=lambda x: x.get('started_at', ''))
        return analyze_reasoning_development(recent_session['session_id'], agent_role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze user virtue development: {str(e)}")

# User Management Endpoints
@app.post("/users")
async def create_user(request: Request):
    """Create a new user"""
    data = await request.json()
    email = data.get("email")
    username = data.get("username", email.split("@")[0] if email else None)

    if not email:
        raise HTTPException(400, "Email required")

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            # Check if user exists
            c.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing = c.fetchone()
            if existing:
                return {"user_id": existing["id"], "message": "User already exists"}

            # Create user
            c.execute("INSERT INTO users (username, email) VALUES (%s, %s) RETURNING id",
                     (username, email))
            user_id = c.fetchone()["id"]
            conn.commit()

    return {"user_id": user_id, "message": "User created"}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    """Get user information"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            if not user:
                raise HTTPException(404, "User not found")
            return dict(user)

@app.post("/sessions")
async def create_session(request: Request):
    """Create a new user session"""
    data = await request.json()
    user_id = data.get("user_id")

    if not user_id:
        raise HTTPException(400, "user_id required")

    import uuid
    session_id = str(uuid.uuid4())

    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO user_sessions (user_id, session_id)
                VALUES (%s, %s)
            """, (user_id, session_id))
            conn.commit()

    return {"session_id": session_id, "user_id": user_id}

@app.get("/users/{user_id}/resurrection-tokens")
def get_user_resurrection_tokens(user_id: int):
    """Get resurrection tokens for a user"""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("""
                SELECT resurrection_token, eiq_value, cycles_completed, created_at,
                       resurrected_at, resurrection_count
                FROM resurrection_tokens
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            tokens = c.fetchall()
            return {"tokens": [dict(token) for token in tokens]}



@app.delete("/streaming-responses/{session_id}")
def clear_streaming_responses(session_id: str):
    """Clear streaming responses for a session."""
    with get_conn() as conn:
        with conn.cursor() as c:
            c.execute("DELETE FROM streaming_responses WHERE session_id = %s", (session_id,))
            conn.commit()
    return {"status": "Streaming responses cleared"}



@app.get("/")
def root():
    return {"status": "nTGT-Ω — PostgreSQL — AMDG"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
