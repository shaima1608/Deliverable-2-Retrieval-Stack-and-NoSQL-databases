from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import cypher_query, seed_graph_from_mongo
from src.ingestion import build_corpus
from src.search import build_bm25_index, hybrid_search, mongo_lookup
from src.stores import get_mongo, seed_mongo, seed_qdrant

# Swagger is moved to /swagger so /docs can be a polished project page.
app = FastAPI(
    title="CSAI415 D2 Retrieval Stack API",
    version="1.0",
    docs_url="/openapi-docs",
    redoc_url="/redoc",
)


def read_stats_safe() -> dict:
    try:
        db = get_mongo()
        return {
            "documents": db[settings.mongo_docs_collection].count_documents({}),
            "chunks": db[settings.mongo_chunks_collection].count_documents({}),
            "mongo_db": settings.mongo_db,
            "mongo_uri": settings.mongo_uri,
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "neo4j_uri": settings.neo4j_uri,
            "status": "online",
        }
    except Exception as e:
        return {
            "documents": "—",
            "chunks": "—",
            "mongo_db": getattr(settings, "mongo_db", "unknown"),
            "mongo_uri": getattr(settings, "mongo_uri", "unknown"),
            "qdrant_url": getattr(settings, "qdrant_url", "unknown"),
            "qdrant_collection": getattr(settings, "qdrant_collection", "paper_chunks"),
            "neo4j_uri": getattr(settings, "neo4j_uri", "unknown"),
            "status": "offline",
            "error": str(e),
        }


def base_page(title: str, body: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg:#030712; --bg2:#06162c; --card:#0f172a; --card2:#111827;
      --muted:#94a3b8; --text:#e5e7eb; --accent:#38bdf8; --accent2:#2563eb;
      --green:#22c55e; --border:#26354d; --warn:#f59e0b; --danger:#ef4444;
      --purple:#a78bfa;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#123a70 0,#081425 38%,#020617 100%); color:var(--text); min-height:100vh; }}
    a {{ color:inherit; }}
    .wrap {{ max-width:1260px; margin:0 auto; padding:28px 20px 70px; }}
    .nav {{ display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:26px; position:sticky; top:0; padding:14px 0; backdrop-filter:blur(14px); z-index:5; }}
    .brand {{ display:flex; align-items:center; gap:12px; font-size:19px; font-weight:900; letter-spacing:.02em; }}
    .logo {{ width:36px; height:36px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg,var(--accent),#14b8a6); color:#00111f; font-weight:1000; box-shadow:0 10px 28px rgba(56,189,248,.25); }}
    .navlinks {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .nav a,.btn,button {{ display:inline-flex; align-items:center; justify-content:center; gap:8px; text-decoration:none; border:1px solid var(--border); background:rgba(15,23,42,.82); color:#dbeafe; padding:11px 16px; border-radius:14px; font-weight:850; cursor:pointer; transition:.18s ease; font-size:15px; }}
    .nav a:hover,.btn:hover,button:hover {{ transform:translateY(-1px); border-color:#3b82f6; box-shadow:0 10px 28px rgba(37,99,235,.22); }}
    button.primary,.btn.primary {{ border:0; background:linear-gradient(135deg,var(--accent),var(--accent2)); color:white; }}
    .grid {{ display:grid; grid-template-columns:1.15fr .85fr; gap:24px; }}
    .panel {{ background:rgba(15,23,42,.93); border:1px solid var(--border); border-radius:26px; box-shadow:0 24px 70px rgba(0,0,0,.35); padding:28px; }}
    .badge {{ display:inline-flex; gap:8px; align-items:center; border:1px solid #164e63; background:#082f49; color:#bae6fd; padding:8px 12px; border-radius:999px; font-size:13px; font-weight:900; }}
    h1 {{ font-size:44px; line-height:1.08; margin:18px 0 12px; letter-spacing:.01em; }}
    h2 {{ margin:0 0 12px; font-size:26px; }}
    h3 {{ margin:0 0 8px; }}
    .sub,.small {{ color:var(--muted); line-height:1.7; }} .small {{ font-size:13px; }}
    input,select,textarea {{ width:100%; background:#07111f; color:var(--text); border:1px solid var(--border); border-radius:15px; padding:14px 15px; font-size:15px; outline:none; }}
    input:focus,select:focus,textarea:focus {{ border-color:#38bdf8; box-shadow:0 0 0 4px rgba(56,189,248,.11); }}
    textarea {{ min-height:136px; font-family:Consolas,monospace; }}
    .row {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .searchbox {{ display:grid; grid-template-columns:1fr 130px; gap:10px; margin-top:20px; }}
    .controls {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }}
    .stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px; }}
    .stat {{ background:#07111f; border:1px solid var(--border); border-radius:19px; padding:18px; min-height:104px; display:flex; flex-direction:column; justify-content:center; overflow:hidden; }}
    .stat b {{ font-size:30px; letter-spacing:.03em; overflow-wrap:anywhere; line-height:1.05; }}
    .stat span {{ display:block; color:var(--muted); font-size:13px; margin-top:7px; }}
    .results {{ margin-top:24px; display:grid; gap:14px; }}
    .result {{ background:rgba(7,17,31,.95); border:1px solid var(--border); border-radius:19px; padding:18px; }}
    .result h3 {{ font-size:17px; color:#f8fafc; }}
    .meta {{ display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 12px; }}
    .chip {{ font-size:12px; color:#cbd5e1; border:1px solid var(--border); border-radius:999px; padding:5px 9px; background:#111827; }}
    .score {{ color:#bbf7d0; border-color:#14532d; background:#052e16; }}
    .text {{ color:#d1d5db; line-height:1.65; }}
    .ok {{ color:#bbf7d0; }} .bad {{ color:#fecaca; }} .warn {{ color:#fde68a; }}
    .err {{ color:#fecaca; background:#450a0a; border:1px solid #7f1d1d; padding:12px; border-radius:14px; margin-top:14px; }}
    .notice {{ background:#082f49; border:1px solid #155e75; color:#bae6fd; padding:12px; border-radius:14px; margin-top:14px; }}
    .success {{ background:#052e16; border:1px solid #14532d; color:#bbf7d0; padding:12px; border-radius:14px; margin-top:14px; }}
    .code {{ background:#020617; border:1px solid var(--border); border-radius:16px; padding:14px; white-space:pre-wrap; overflow:auto; font-family:Consolas,monospace; color:#d1d5db; max-height:540px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:16px; }} td,th {{ border-bottom:1px solid var(--border); padding:13px; text-align:left; vertical-align:top; }} th {{ color:#bae6fd; width:260px; }}
    .endpoint {{ display:grid; grid-template-columns:92px 1fr auto; gap:12px; align-items:center; padding:14px; border:1px solid var(--border); border-radius:16px; background:#07111f; margin:10px 0; }}
    .method {{ font-weight:1000; text-align:center; padding:7px 9px; border-radius:10px; }}
    .get {{ background:#1d4ed8; }} .post {{ background:#047857; }}
    .footer {{ margin-top:22px; color:#64748b; font-size:12px; }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} h1 {{ font-size:34px; }} .stats,.controls,.searchbox {{ grid-template-columns:1fr; }} .nav {{ align-items:flex-start; flex-direction:column; }} .endpoint {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <div class="nav">
      <a class="brand" href="/"><div class="logo">D2</div><span>CSAI415 Retrieval Stack</span></a>
      <div class="navlinks">
        <a href="/">Search UI</a>
        <a href="/docs">API Docs</a>
        <a href="/stats">Stats</a>
        <a href="/health">Health</a>
      </div>
    </div>
    {body}
  </main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def professional_ui():
    stats = read_stats_safe()
    status_box = "success" if stats.get("status") == "online" else "err"
    body = f"""
    <section class="grid">
      <div class="panel">
        <div class="badge">🔎 Hybrid Retrieval + Graph Build</div>
        <h1>CSAI415 D2 Research Paper Search</h1>
        <p class="sub">Search your PDF corpus using BM25 + dense Qdrant retrieval, with MongoDB provenance and Neo4j graph support.</p>
        <div class="searchbox">
          <input id="query" placeholder="Ask about Adam optimization, RAG, word vectors, reinforcement learning..." value="What is Adam optimization?" />
          <button class="primary" onclick="runSearch()">Search</button>
        </div>
        <div class="controls">
          <select id="topk"><option value="5">Top 5 results</option><option value="3">Top 3 results</option><option value="10">Top 10 results</option></select>
          <select id="alpha"><option value="">Default hybrid weight</option><option value="0.3">More BM25</option><option value="0.55">Balanced</option><option value="0.8">More dense</option></select>
        </div>
        <div class="row" style="margin-top:16px">
          <button onclick="setExample('What is Adam optimization?')">Adam</button>
          <button onclick="setExample('What are word vectors?')">Word vectors</button>
          <button onclick="setExample('What is retrieval augmented generation?')">RAG</button>
        </div>
        <div id="searchStatus" class="small" style="margin-top:14px">Ready to search.</div>
      </div>
      <div class="panel">
        <h2>System status</h2>
        <p class="small">Live counts from your local stores.</p>
        <div class="stats">
          <div class="stat"><b id="docs">{stats['documents']}</b><span>Documents</span></div>
          <div class="stat"><b id="chunks">{stats['chunks']}</b><span>Chunks</span></div>
          <div class="stat"><b id="collection">{stats['qdrant_collection']}</b><span>Qdrant collection</span></div>
        </div>
        <div id="storeStatus" class="{status_box}">Store status: {stats['status']}</div>
        <div class="row" style="margin-top:18px">
          <a class="btn" href="/docs">API Docs</a>
          <a class="btn" href="/stats">Stats</a>
          <a class="btn" href="/health">Health</a>
        </div>
      </div>
    </section>
    <section id="results" class="results"></section>
<script>
function escapeHtml(x){{return String(x??'').replace(/[&<>'"]/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[m]))}}
function setExample(q){{document.getElementById('query').value=q; runSearch();}}
async function loadStats(){{
  try{{
    const r=await fetch('/stats-json');
    if(!r.ok) throw new Error(await r.text());
    const s=await r.json();
    document.getElementById('docs').textContent=s.documents ?? '—';
    document.getElementById('chunks').textContent=s.chunks ?? '—';
    document.getElementById('collection').textContent=s.qdrant_collection ?? '—';
    document.getElementById('storeStatus').className='success';
    document.getElementById('storeStatus').textContent='Store status: online';
  }}catch(e){{
    document.getElementById('storeStatus').className='err';
    document.getElementById('storeStatus').textContent='Stats could not load: '+e.message;
  }}
}}
async function runSearch(){{
  const query=document.getElementById('query').value.trim();
  if(query.length < 2){{ document.getElementById('searchStatus').innerHTML='<div class="err">Please enter a question.</div>'; return; }}
  const top_k=parseInt(document.getElementById('topk').value);
  const alphaVal=document.getElementById('alpha').value;
  const body={{query,top_k}};
  if(alphaVal) body.alpha=parseFloat(alphaVal);
  document.getElementById('searchStatus').innerHTML='Searching indexed papers...';
  document.getElementById('results').innerHTML='';
  try{{
    const r=await fetch('/search',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
    if(!r.ok) throw new Error(await r.text());
    const data=await r.json();
    const items=data.results || [];
    document.getElementById('searchStatus').innerHTML='Returned '+items.length+' results in '+(data.latency_ms ?? '—')+' ms.';
    document.getElementById('results').innerHTML=items.map((it,i)=>{{
      const title=escapeHtml(it.title||it.filename||it.paper_id||'Retrieved chunk');
      const text=escapeHtml(it.text||'');
      const score=Number(it.score ?? 0).toFixed(4);
      const bm25=Number(it.bm25_score ?? 0).toFixed(4);
      const dense=Number(it.dense_score ?? 0).toFixed(4);
      const citation=escapeHtml(it.citation || `${{it.filename || ''}} p.${{it.page_start || ''}}-${{it.page_end || ''}}`);
      const chunk=escapeHtml(it.chunk_id || '');
      return `<article class="result"><h3>${{i+1}}. ${{title}}</h3><div class="meta"><span class="chip score">Hybrid: ${{score}}</span><span class="chip">BM25: ${{bm25}}</span><span class="chip">Dense: ${{dense}}</span><span class="chip">${{citation}}</span><span class="chip">${{chunk}}</span></div><div class="text">${{text}}</div></article>`;
    }}).join('') || '<div class="result">No results found.</div>';
  }}catch(e){{ document.getElementById('searchStatus').innerHTML='<div class="err">'+escapeHtml(e.message)+'</div>'; }}
}}
loadStats();
</script>
    """
    return base_page("CSAI415 D2 Retrieval Stack", body)


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def api_docs_page():
    body = """
    <section class="panel">
      <div class="badge">📘 Professional API Docs</div>
      <h1>D2 Retrieval Stack API</h1>
      <p class="sub">Clean documentation page for the demo. Use the buttons below to test endpoints or open Swagger for the automatic OpenAPI interface.</p>
      <div class="row" style="margin:18px 0 24px">
        <a class="btn primary" href="/swagger">Open API Tester</a>
        <a class="btn" href="/">Back to Search UI</a>
      </div>
      <div class="endpoint"><span class="method get">GET</span><div><b>/</b><div class="small">Professional search interface.</div></div><a class="btn" href="/">Open</a></div>
      <div class="endpoint"><span class="method post">POST</span><div><b>/search</b><div class="small">Hybrid BM25 + dense Qdrant retrieval. Body: query, top_k, alpha.</div></div><a class="btn" href="/swagger">Test</a></div>
      <div class="endpoint"><span class="method post">POST</span><div><b>/cypher</b><div class="small">Run Neo4j Cypher queries for graph inspection.</div></div><a class="btn" href="/swagger">Test</a></div>
      <div class="endpoint"><span class="method get">GET</span><div><b>/stats</b><div class="small">Readable stats dashboard.</div></div><a class="btn" href="/stats">Open</a></div>
      <div class="endpoint"><span class="method get">GET</span><div><b>/health</b><div class="small">Readable health check page.</div></div><a class="btn" href="/health">Open</a></div>
      <div class="endpoint"><span class="method get">GET</span><div><b>/mongo/chunk/{chunk_id}</b><div class="small">Look up a single chunk from MongoDB.</div></div><a class="btn" href="/swagger">Test</a></div>
    </section>
    """
    return base_page("D2 Professional API Docs", body)


@app.get("/openapi-json", response_class=HTMLResponse, include_in_schema=False)
def openapi_json_page():
    import json
    spec = app.openapi()
    body = f"""
    <section class="panel">
      <div class="badge">🧾 OpenAPI Schema</div>
      <h1>Formatted OpenAPI JSON</h1>
      <p class="sub">This page shows the API schema in a readable format. It is useful when the instructor wants to see the technical structure behind the endpoints.</p>
      <div class="row" style="margin:18px 0 22px">
        <a class="btn" href="/swagger">Back to API Tester</a>
        <a class="btn" href="/docs">API Docs</a>
        <a class="btn" href="/openapi.json" target="_blank">Open Raw JSON</a>
      </div>
      <pre class="code">{json.dumps(spec, indent=2)}</pre>
    </section>
    """
    return base_page("D2 OpenAPI JSON", body)



def render_tester_result(title: str, value) -> str:
    import html
    import json
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        text = str(value)
    return f"""
      <h2 style="margin-top:22px">Response preview</h2>
      <pre class="code">{html.escape(title)}\n\n{html.escape(text)}</pre>
    """


def swagger_tester_body(response_html: str = "") -> str:
    import html
    node_counts = "MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY count DESC"
    papers_topics = "MATCH (p:Paper)-[:ABOUT]->(t:Topic) RETURN p.title AS paper, collect(t.name) AS topics LIMIT 5"
    default_cypher = html.escape(node_counts)
    return f"""
    <section class="panel">
      <div class="badge">🧪 Friendly API Tester</div>
      <h1>D2 Endpoint Tester</h1>
      <p class="sub">Use this page during the demo to test retrieval, Neo4j graph queries, and system checks. This version uses normal links and forms, so it works even if browser JavaScript is blocked.</p>
      <div class="row" style="margin:18px 0 24px">
        <a class="btn" href="/">Search UI</a>
        <a class="btn" href="/docs">API Docs</a>
        <a class="btn" href="/stats">Stats</a>
        <a class="btn" href="/health">Health</a>
      </div>

      <div class="grid">
        <form class="result" method="get" action="/swagger-search">
          <h2>1. Test /search</h2>
          <p class="small">Runs the hybrid retrieval endpoint. It searches BM25 + Qdrant and returns cited chunks from MongoDB.</p>
          <label class="small" for="query">Question</label>
          <input name="query" id="query" value="What is Adam optimization?" />
          <div class="controls">
            <div><label class="small" for="top_k">Top K</label><input name="top_k" id="top_k" type="number" min="1" max="20" value="5" /></div>
            <div><label class="small" for="alpha">Hybrid weight alpha</label><input name="alpha" id="alpha" value="0.55" /></div>
          </div>
          <div class="row" style="margin-top:14px">
            <a class="btn" href="/swagger-search?query=What+is+Adam+optimization%3F&top_k=5&alpha=0.55">Adam</a>
            <a class="btn" href="/swagger-search?query=What+are+word+vectors%3F&top_k=5&alpha=0.55">Word vectors</a>
            <a class="btn" href="/swagger-search?query=What+is+retrieval+augmented+generation%3F&top_k=5&alpha=0.55">RAG</a>
            <button class="primary" type="submit">Run Search</button>
          </div>
        </form>

        <form class="result" method="get" action="/swagger-cypher">
          <h2>2. Test /cypher</h2>
          <p class="small">Runs a Neo4j graph query to prove the graph build for Papers, Authors, Topics, and Venues.</p>
          <label class="small" for="cypher_query">Cypher query</label>
          <textarea name="query" id="cypher_query">{default_cypher}</textarea>
          <div class="row" style="margin-top:14px">
            <a class="btn" href="/swagger-cypher?query=MATCH+%28n%29+RETURN+labels%28n%29+AS+labels%2C+count%28n%29+AS+count+ORDER+BY+count+DESC">Node counts</a>
            <a class="btn" href="/swagger-cypher?query=MATCH+%28p%3APaper%29-%5B%3AABOUT%5D-%3E%28t%3ATopic%29+RETURN+p.title+AS+paper%2C+collect%28t.name%29+AS+topics+LIMIT+5">Papers + topics</a>
            <button class="primary" type="submit">Run Cypher</button>
          </div>
        </form>
      </div>

      <div class="result" style="margin-top:18px">
        <h2>3. System checks</h2>
        <p class="small">These links open the clean project pages.</p>
        <div class="row">
          <a class="btn" href="/health">Check Health</a>
          <a class="btn" href="/stats">Check Stats</a>
          <a class="btn" href="/openapi-json">Show OpenAPI JSON</a>
          <a class="btn" href="/openapi-docs" target="_blank">Technical Swagger</a>
        </div>
      </div>

      {response_html or '<h2 style="margin-top:22px">Response preview</h2><pre class="code">Ready. Click a button or submit a form above.</pre>'}
    </section>
    """


@app.get("/swagger", response_class=HTMLResponse, include_in_schema=False)
def friendly_swagger_tester():
    return base_page("D2 Friendly API Tester", swagger_tester_body())


@app.get("/swagger-search", response_class=HTMLResponse, include_in_schema=False)
def swagger_search(query: str = "What is Adam optimization?", top_k: int = 5, alpha: float | None = 0.55):
    try:
        result = hybrid_search(query, top_k=top_k, alpha=alpha)
        response_html = render_tester_result(f"/search response for: {query}", result)
    except Exception as e:
        response_html = render_tester_result("/search error", {"error": str(e)})
    return base_page("D2 Friendly API Tester", swagger_tester_body(response_html))


@app.get("/swagger-cypher", response_class=HTMLResponse, include_in_schema=False)
def swagger_cypher(query: str = "MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY count DESC"):
    try:
        result = {"rows": cypher_query(query, {})}
        response_html = render_tester_result("/cypher response", result)
    except Exception as e:
        response_html = render_tester_result("/cypher error", {"error": str(e), "query": query})
    return base_page("D2 Friendly API Tester", swagger_tester_body(response_html))


@app.get("/api-console", response_class=HTMLResponse, include_in_schema=False)
def api_console():
    body = """
    <section class="panel">
      <div class="badge">🧪 API Console</div>
      <h1>Test D2 Endpoints</h1>
      <p class="sub">Use this page for demo testing without typing JSON in Swagger.</p>
      <div class="grid">
        <div class="result">
          <h2>Hybrid Search</h2>
          <input id="q" value="What is Adam optimization?" />
          <div class="controls"><input id="k" type="number" value="5" min="1" max="20" /><input id="a" value="0.55" /></div>
          <button class="primary" style="margin-top:10px" onclick="postSearch()">Run /search</button>
        </div>
        <div class="result">
          <h2>Neo4j Cypher</h2>
          <textarea id="cy">MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY count DESC</textarea>
          <button class="primary" style="margin-top:10px" onclick="postCypher()">Run /cypher</button>
        </div>
      </div>
      <h2 style="margin-top:22px">Response</h2>
      <pre id="out" class="code">Waiting for request...</pre>
    </section>
<script>
async function renderResponse(r){
  const text = await r.text();
  try { document.getElementById('out').textContent = JSON.stringify(JSON.parse(text), null, 2); }
  catch(e) { document.getElementById('out').textContent = text; }
}
async function postSearch(){
  const alphaText=document.getElementById('a').value.trim();
  const body={query:document.getElementById('q').value, top_k:Number(document.getElementById('k').value)};
  if(alphaText) body.alpha=Number(alphaText);
  const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  await renderResponse(r);
}
async function postCypher(){
  const body={query:document.getElementById('cy').value, parameters:{}};
  const r=await fetch('/cypher',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  await renderResponse(r);
}
</script>
    """
    return base_page("D2 API Console", body)


@app.get("/stats", response_class=HTMLResponse, include_in_schema=False)
def stats_page():
    s = read_stats_safe()
    status_class = "success" if s.get("status") == "online" else "err"
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in s.items())
    body = f"""
    <section class="panel">
      <div class="badge">📊 Stats Dashboard</div>
      <h1>System Statistics</h1>
      <p class="sub">Readable dashboard for MongoDB and Qdrant configuration.</p>
      <div class="stats">
        <div class="stat"><b>{s.get('documents')}</b><span>Documents in MongoDB</span></div>
        <div class="stat"><b>{s.get('chunks')}</b><span>Chunks in MongoDB</span></div>
        <div class="stat"><b>{s.get('qdrant_collection')}</b><span>Qdrant collection</span></div>
      </div>
      <div class="{status_class}">Store status: {s.get('status')}</div>
      <table><tbody>{rows}</tbody></table>
      <div class="row" style="margin-top:18px"><a class="btn" href="/">Back to Search</a><a class="btn" href="/docs">API Docs</a><a class="btn" href="/health">Health</a></div>
    </section>
    """
    return base_page("D2 Stats Dashboard", body)


@app.get("/health", response_class=HTMLResponse, include_in_schema=False)
def health_page():
    s = read_stats_safe()
    online = s.get("status") == "online"
    health_class = "ok" if online else "bad"
    box_class = "success" if online else "err"
    body = f"""
    <section class="panel">
      <div class="badge">✅ Health Check</div>
      <h1>System Health</h1>
      <div class="result"><h2 class="{health_class}">Overall status: {'ONLINE' if online else 'OFFLINE'}</h2><p class="small">FastAPI is running. MongoDB status is checked live from the local store.</p></div>
      <div class="{box_class}">{'All main services required for search are available.' if online else 'FastAPI is running, but one or more stores could not be reached.'}</div>
      <table><tbody>
        <tr><th>FastAPI</th><td class="ok">running</td></tr>
        <tr><th>MongoDB</th><td>{s.get('status')}</td></tr>
        <tr><th>Documents</th><td>{s.get('documents')}</td></tr>
        <tr><th>Chunks</th><td>{s.get('chunks')}</td></tr>
        <tr><th>Qdrant collection</th><td>{s.get('qdrant_collection')}</td></tr>
        <tr><th>Mongo URI</th><td>{s.get('mongo_uri')}</td></tr>
      </tbody></table>
      <div class="row" style="margin-top:18px"><a class="btn" href="/">Back to Search</a><a class="btn" href="/docs">API Docs</a><a class="btn" href="/stats">Stats</a></div>
    </section>
    """
    return base_page("D2 Health Check", body)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = 5
    alpha: float | None = None


class CypherRequest(BaseModel):
    query: str
    parameters: dict = {}


@app.get("/health-json", include_in_schema=False)
def health_json():
    s = read_stats_safe()
    return {"status": "ok" if s.get("status") == "online" else "degraded", "stores": s.get("status"), "documents": s.get("documents"), "chunks": s.get("chunks"), "qdrant_collection": s.get("qdrant_collection")}


@app.get("/stats-json", include_in_schema=False)
def stats_json():
    s = read_stats_safe()
    if s.get("status") == "offline":
        raise HTTPException(status_code=500, detail=s.get("error", "stats unavailable"))
    return {
        "documents": s["documents"],
        "chunks": s["chunks"],
        "mongo_db": s["mongo_db"],
        "qdrant_collection": s["qdrant_collection"],
    }


@app.post("/ingest")
def ingest():
    try:
        docs, chunks = build_corpus(settings.pdf_dir, settings.chunk_size, settings.chunk_overlap)
        mongo_info = seed_mongo(docs, chunks)
        qdrant_info = seed_qdrant(chunks)
        bm25_path = build_bm25_index(chunks)
        graph_info = seed_graph_from_mongo()
        return {"mongo": mongo_info, "qdrant": qdrant_info, "bm25_index": bm25_path, "neo4j": graph_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    try:
        return hybrid_search(req.query, top_k=req.top_k, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mongo/chunk/{chunk_id}")
def chunk(chunk_id: str):
    out = mongo_lookup(chunk_id)
    if not out:
        raise HTTPException(status_code=404, detail="chunk not found")
    return out


@app.post("/cypher")
def cypher(req: CypherRequest):
    try:
        return {"rows": cypher_query(req.query, req.parameters)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
