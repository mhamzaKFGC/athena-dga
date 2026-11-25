"""Check whether the domains in athena_synthetic.csv look registered.
Read-only: one public DNS query (NS) per domain, then one RDAP lookup (rdap.org) for domains DNS says do not exist.
Run: python check_registered.py
Result meaning:
  registered  - the DNS has name servers, or RDAP found a registration record
  likely free - DNS NXDOMAIN and RDAP says 404 (no record). Still confirm at a registrar before buying.
  unknown     - lookup failed or was inconclusive"""
import csv
import json
import socket
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def dns_ns(domain, server="1.1.1.1"):
    """Return 'NOERROR_NS', 'NOERROR_NODATA', 'NXDOMAIN' or 'ERROR' for an NS query over UDP."""
    q = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
    for part in domain.split("."):
        q += bytes([len(part)]) + part.encode()
    q += b"\x00\x00\x02\x00\x01"  # type NS, class IN
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(4)
    try:
        s.sendto(q, (server, 53))
        r = s.recv(2048)
    except OSError:
        return "ERROR"
    finally:
        s.close()
    rcode = r[3] & 0x0F
    answers = struct.unpack(">H", r[6:8])[0]
    if rcode == 3:
        return "NXDOMAIN"
    if rcode == 0:
        return "NOERROR_NS" if answers else "NOERROR_NODATA"
    return "ERROR"


def rdap(domain):
    """Return 'registered', 'free' or 'unknown' from rdap.org."""
    req = urllib.request.Request(f"https://rdap.org/domain/{domain}", headers={"User-Agent": "domain-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.load(resp)
            return "registered"
    except urllib.error.HTTPError as e:
        return "free" if e.code == 404 else "unknown"
    except (OSError, ValueError):
        return "unknown"


rows = list(csv.DictReader((HERE / "athena_synthetic.csv").open(encoding="utf-8")))
out = []
print(f"{'domain':32}{'DNS':18}{'RDAP':12}verdict")
for r in rows:
    d = r["domain"]
    dns = dns_ns(d)
    rd = "-"
    if dns == "NOERROR_NS":
        verdict = "registered"
    else:
        rd = rdap(d)
        time.sleep(0.5)
        verdict = {"registered": "registered", "free": "likely free"}.get(rd, "unknown")
    out.append((d, dns, rd, verdict))
    print(f"{d:32}{dns:18}{rd:12}{verdict}")

with (HERE / "registration_check.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["domain", "dns", "rdap", "verdict"])
    w.writerows(out)
print("saved registration_check.csv")
