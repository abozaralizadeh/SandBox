import sys

# Remove Azure App Service helper directory that ships outdated stdlib shims
sys.path = [p for p in sys.path if not p.startswith("/agents/python")]
