"""Turn the repo's Actions secrets into environment variables for the later steps.

The generators read their settings from the environment, the same names the App Service
had. GitHub stores secret names in UPPER case, but several of ours are lower case
(`connection_string`, `aiblog_table_name`, ...), so the original spelling comes from the
`SANDBOX_ENV_NAMES` variable that scripts/sync_github_secrets.sh writes next to the secrets.

    SECRETS_JSON='${{ toJSON(secrets) }}' ENV_NAMES='${{ vars.SANDBOX_ENV_NAMES }}' \
        python3 jobs/export_env.py
"""
import json
import os
import secrets
import sys


def main() -> int:
    values = json.loads(os.environ.get("SECRETS_JSON") or "{}")
    names = [n.strip() for n in (os.environ.get("ENV_NAMES") or "").split(",") if n.strip()]
    spelling = {n.upper(): n for n in names}
    target = os.environ["GITHUB_ENV"]
    count = 0
    with open(target, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            if key == "github_token" or key.upper() == "GITHUB_TOKEN":
                continue
            name = spelling.get(key.upper(), key)
            marker = f"EOF_{secrets.token_hex(8)}"
            fh.write(f"{name}<<{marker}\n{value}\n{marker}\n")
            count += 1
    print(f"[export_env] exported {count} settings")
    missing = [n for n in names if n.upper() not in {k.upper() for k in values}]
    if missing:
        print(f"[export_env] listed but not set as secrets: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
