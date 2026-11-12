"""
ssrf_protection.py - Server-Side Request Forgery (SSRF) Protection Engine
=========================================================================
Strictly validates outbound webhook destination URLs to prevent SSRF attacks against:
- Private RFC 1918 subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Loopback addresses (127.0.0.0/8, ::1)
- Link-local and cloud metadata endpoints (169.254.169.254, 169.254.0.0/16, fe80::/10)
- Multicast, broadcast, and reserved network segments
- Non-HTTP/HTTPS URI schemes and dangerous internal ports
"""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple, List

from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException
from backend.app.core.logging import get_logger

logger = get_logger("core.security.ssrf")

# Disallowed IP Networks
DISALLOWED_NETWORKS = [
    # IPv4 loopback, private, link-local, multicast, broadcast
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),         # Loopback
    ipaddress.ip_network("169.254.0.0/16"),       # Link-local & cloud metadata (169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),       # Private RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),         # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),         # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),      # Private RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),       # Benchmark testing
    ipaddress.ip_network("198.51.100.0/24"),     # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),      # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved / Future use
    ipaddress.ip_network("255.255.255.255/32"),  # Limited broadcast
    # IPv6 loopback, link-local, unique-local, multicast
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("::/128"),              # Unspecified
    ipaddress.ip_network("64:ff9b::/96"),        # IPv4/IPv6 translation
    ipaddress.ip_network("100::/64"),            # Discard-only prefix
    ipaddress.ip_network("2001:db8::/32"),       # Documentation
    ipaddress.ip_network("fc00::/7"),            # Unique local address (ULA)
    ipaddress.ip_network("fe80::/10"),           # Link-local
    ipaddress.ip_network("ff00::/8"),            # Multicast
]

# Prohibited internal service ports
BLOCKED_PORTS = {
    21,    # FTP
    22,    # SSH
    23,    # Telnet
    25,    # SMTP
    110,   # POP3
    143,   # IMAP
    445,   # SMB
    1433,  # MSSQL
    1521,  # Oracle
    2375,  # Docker
    2376,  # Docker SSL
    2379,  # etcd
    3306,  # MySQL
    5432,  # PostgreSQL
    6379,  # Redis
    9200,  # Elasticsearch
    11211, # Memcached
    27017, # MongoDB
}


class SSRFValidationError(ValidationException):
    """Raised when an outbound URL fails SSRF safety validation."""
    pass


def is_ip_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Checks whether an IP address belongs to any private or reserved network."""
    # Check standard library properties first
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True

    # Check explicit network definitions
    for net in DISALLOWED_NETWORKS:
        if ip in net:
            return True

    return False


is_private_or_reserved_ip = is_ip_disallowed


def validate_webhook_url(
    url: str,
    allow_insecure: bool = False,
    allow_http: Optional[bool] = None
) -> Tuple[bool, str]:
    """
    Validates a webhook target URL against SSRF rules.
    Returns (is_valid, sanitized_url). Raises SSRFValidationError if invalid.
    """
    if not url or not isinstance(url, str):
        raise SSRFValidationError("Webhook URL must be a non-empty string.")

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise SSRFValidationError(f"Malformed URL structure: {str(exc)}")

    # 1. Scheme Validation
    effective_allow_http = allow_http if allow_http is not None else (allow_insecure or settings.WEBHOOK_ALLOW_INSECURE_HTTP)
    scheme = (parsed.scheme or "").lower()

    if scheme not in ("http", "https"):
        raise SSRFValidationError(f"Invalid URL scheme '{scheme}'. Only HTTPS URLs are allowed.")

    if scheme == "http" and not effective_allow_http:
        raise SSRFValidationError("Insecure HTTP webhooks are prohibited in production. HTTPS is required.")

    # 2. Hostname Validation
    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError("Webhook URL must include a valid hostname.")

    clean_host = hostname.lower().strip("[]")
    if clean_host in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback") or clean_host.endswith(".local"):
        raise SSRFValidationError(f"Targeting localhost or local domain hostnames ('{hostname}') is prohibited.")

    # 3. Port Validation
    port = parsed.port
    if port is not None:
        if port in BLOCKED_PORTS:
            raise SSRFValidationError(f"Targeting port {port} is blocked or non-standard by SSRF policy.")
        if port < 1 or port > 65535:
            raise SSRFValidationError(f"Port {port} is out of valid range (1-65535).")

    # 4. Direct IP or DNS Resolution Check
    try:
        # Check if hostname is directly an IP address
        direct_ip = ipaddress.ip_address(clean_host)
        if is_ip_disallowed(direct_ip):
            raise SSRFValidationError(f"Targeting loopback or private IP address ({direct_ip}) is prohibited.")
        return True, url
    except ValueError:
        # Hostname is a domain name, proceed to DNS resolution
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        resolved_ips = set()
        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            resolved_ips.add(ipaddress.ip_address(ip_str))

        if not resolved_ips:
            raise SSRFValidationError(f"Failed to resolve hostname '{hostname}'.")

        for ip in resolved_ips:
            if is_ip_disallowed(ip):
                logger.warning(f"SSRF violation: Hostname '{hostname}' resolved to prohibited IP {ip}")
                raise SSRFValidationError(f"Hostname '{hostname}' resolves to loopback or private IP address ({ip}).")

    except socket.gaierror as exc:
        raise SSRFValidationError(f"DNS resolution failed for hostname '{hostname}': {str(exc)}")
    except SSRFValidationError:
        raise
    except Exception as exc:
        raise SSRFValidationError(f"Failed verifying IP destination for '{hostname}': {str(exc)}")

    return True, url
