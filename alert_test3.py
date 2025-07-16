
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 2. ALERT_TYPE_MAPPING: maps alert "type" codes to names and display names
ALERT_TYPE_MAPPING = {
    19100: {"name": "AlertXtendCpuUsageHigh", "display_name": "Device CPU Usage High"},
    19200: {"name": "AlertXtendMemUsageHigh", "display_name": "Device Memory Usage High"},
    19300: {"name": "AlertXtendWifiSignalLow", "display_name": "Device WiFi Signal Strength Low"},
    19400: {"name": "AlertXtendSoftwareVulnerability", "display_name": "Installed Software Vulnerability"},
    19500: {"name": "AlertXtendUnauthorizedSsid", "display_name": "Unauthorized SSID Connection"},
    19600: {"name": "AlertXtendUnauthorizedDns", "display_name": "Unauthorized DNS Connection"},
    19700: {"name": "AlertXtendSigma", "display_name": "Sigma Signature Detected on Device"},
    19800: {"name": "AlertXtendUnauthorizedDomain", "display_name": "Unauthorized Domain Connection"},
    19900: {"name": "AlertXtendProvena", "display_name": "AI-Provenance"},
    21000: {"name": "AlertRtdHighGeneric", "display_name": "Round Trip Delay High"},
    21001: {"name": "AlertAvbwLowGeneric", "display_name": "Available Bandwidth Low"},
    21002: {"name": "AlertJitterHighGeneric", "display_name": "Jitter High"},
    21003: {"name": "AlertLossHighGeneric", "display_name": "Loss High"},
    21100: {"name": "AlertDnsLookupHigh", "display_name": "DNS Lookup High"},
    21101: {"name": "AlertDnsSuccessAttemptRatio", "display_name": "DNS Success Attempt Ratio"},
    21200: {"name": "AlertHttpConnectHigh", "display_name": "HTTP Connect High"},
    21201: {"name": "AlertHttpFetchHigh", "display_name": "HTTP Fetch High"},
    21202: {"name": "AlertHttpRedirectHigh", "display_name": "HTTP Redirect High"},
    21203: {"name": "AlertHttpResponseHigh", "display_name": "HTTP Response High"},
    21204: {"name": "AlertHttpSslHigh", "display_name": "HTTP SSL High"},
    21300: {"name": "AlertTcpRttHigh", "display_name": "TCP RTT High"},
    21301: {"name": "AlertTcpClientRetransHigh", "display_name": "TCP Client Retrans High"},
    21302: {"name": "AlertTcpServerRetransHigh", "display_name": "TCP Server Retrans High"},
    21303: {"name": "AlertUdpClntToSrvrLossHigh", "display_name": "UDP Client to Server Loss High"},
    21304: {"name": "AlertUdpRttHigh", "display_name": "UDP RTT High"},
    21305: {"name": "AlertUdpSrvrToClntLossHigh", "display_name": "UDP Server to Client Loss High"},
    21306: {"name": "AlertUdpClntToSrvrJitterHigh", "display_name": "UDP Client to Server Jitter High"},
    21307: {"name": "AlertUdpSrvrToClntJitterHigh", "display_name": "UDP Server to Client Jitter High"},
    21400: {"name": "FlowUploadingHigh", "display_name": "Flow Uploading High"},
    21401: {"name": "FlowDownloadingHigh", "display_name": "Flow Downloading High"},
    21402: {"name": "FlowScanning", "display_name": "Flow Scanning"},
    22000: {"name": "AlertThreatDnsClientTotalCountHigh", "display_name": "Threat DNS Client Total Count High"},
    22001: {"name": "AlertThreatDnsClientServerLongestStreakHigh", "display_name": "Threat DNS Client Server Longest Streak High"},
    22002: {"name": "AlertThreatDnsClientServerTotalChurnHigh", "display_name": "Threat DNS Client Server Total Churn High"},
    22003: {"name": "AlertThreatDnsClientQueryLongestStreakHigh", "display_name": "Threat DNS Client Query Longest Streak High"},
    22004: {"name": "AlertThreatDnsClientQueryTotalChurnHigh", "display_name": "Threat DNS Client Query Total Churn High"}
}

# 3. Load and flatten MITRE ATT&CK Enterprise JSON
attack_path = Path("enterprise-attack.json")
attack_data = json.loads(attack_path.read_text(encoding="utf-8"))
techniques = [
    {
        "tid": obj["technique_id"],
        "name": obj["name"],
        "description": obj.get("description", ""),
        "tactics": obj.get("tactics", [])
    }
    for obj in attack_data["objects"]
    if obj.get("framework") == "MITRE ATT&CK"
]

# 4. Build BM25 index
corpus = [(t["name"] + " " + t["description"]).split() for t in techniques]
bm25 = BM25Okapi(corpus)

# 5. Build FAISS embedding index
model = SentenceTransformer("all-MiniLM-L6-v2")
descs = [t["description"] for t in techniques]
embeddings = model.encode(descs, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings)

# 6. Load alerts and apply processing limit
LOG_LIMIT = 100  # max number of alerts to process (None for no limit)
alerts = json.loads(Path("es_hits.json").read_text(encoding="utf-8"))["hits"]["hits"]
if LOG_LIMIT is not None:
    alerts = alerts[:LOG_LIMIT]

# 7. Build combined query from prioritized fields

def build_query(alert):
    parts = []
    # 7.1 Context from type
    t = alert.get("type")
    if t in ALERT_TYPE_MAPPING:
        m = ALERT_TYPE_MAPPING[t]
        parts.append(f"Type: {m["name"]} ({m["display_name"]})")
    # 7.2 Message
    msg = alert.get("msg")
    if msg:
        parts.append(f"Message: {msg}")
    # 7.3 Severity label
    sev_map = {1: "low", 2: "medium", 3: "high"}
    sev = alert.get("severity")
    if sev in sev_map:
        parts.append(f"Severity: {sev_map[sev]}")
    # 7.4 Anomaly sentences
    anomalies = alert.get("metadata", {}).get("anomalySentences", [])
    if anomalies:
        parts.append("Anomalies: " + " ".join(anomalies))
    # 7.5 Sigma alerts tags
    sigma = alert.get("metadata", {}).get("sigmaAlerts", {})
    if sigma:
        tags = []
        for alerts_list in sigma.values():
            for entry in alerts_list:
                tags.extend(entry.get("tags", []))
        if tags:
            parts.append("SigmaTags: " + " ".join(tags))
    # 7.6 Unauthorized domains
    ud = alert.get("metadata", {}).get("unauthorizedDomains", {})
    if ud:
        domains = [d for doms in ud.values() for d in doms]
        parts.append("Domains: " + " ".join(domains))
    # 7.7 Unauthorized DNS servers
    uds = alert.get("metadata", {}).get("unauthorizedDnsServers", {})
    if uds:
        dns = [s for srvs in uds.values() for s in srvs]
        parts.append("DnsServers: " + " ".join(dns))
    return " ".join(parts)

# 8. Map alerts to top-K TTPs

def map_to_ttps(alert, top_k=3):
    query_text = build_query(alert)
    # 8.1 BM25 scoring
    tokens = query_text.split()
    bm_scores = bm25.get_scores(tokens)
    # 8.2 Embedding scoring
    q_emb = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, top_k)
    emb_scores = {idx: float(D[0][i]) for i, idx in enumerate(I[0])}
    # 8.3 Combine and rank
    combined = [(bm_scores[i] + emb_scores.get(i, 0.0), techniques[i]) for i in range(len(techniques))]
    combined.sort(key=lambda x: x[0], reverse=True)
    return [
        {"tid": tech["tid"], "name": tech["name"], "tactics": tech.get("tactics", []), "score": round(score,4)}
        for score, tech in combined[:top_k]
    ]

# 9. Execute mapping and write output
mapped = []
for hit in alerts:
    alert_src = hit.get("_source", {})
    suggestions = map_to_ttps(alert_src)
    mapped.append({
        "id": hit.get("_id"),
        "type": alert_src.get("type"),
        "suggestions": suggestions
    })

# 10. Write final JSON output
out_file = Path("mapped_enterprise_ttps.json")
out_file.write_text(json.dumps(mapped, indent=2))

# 11. Optional: print a sample mapping
if mapped:
    import pprint; pprint.pprint(mapped[0])
