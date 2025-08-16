import argparse
import sys
from datetime import datetime

# Reuse existing DB helpers from the app
from main import init_auth_db, get_user_by_username, create_user_db, update_user_db


def main():
    parser = argparse.ArgumentParser(description="Manage users (create/update) in auth.db")
    parser.add_argument("--username", required=True, help="Username to create/update")
    parser.add_argument("--password", required=True, help="Plain password")
    parser.add_argument("--role", default="admin", choices=["admin", "readonly", "user"], help="User role")
    parser.add_argument("--activate", action="store_true", help="Mark user as active")
    parser.add_argument("--deactivate", action="store_true", help="Mark user as inactive")
    args = parser.parse_args()

    if args.activate and args.deactivate:
        print("ERR: --activate and --deactivate cannot be used together", file=sys.stderr)
        return 2

    init_auth_db()
    username = args.username.strip()
    password = args.password.strip()
    role = args.role.strip() or "admin"
    is_active = True
    if args.deactivate:
        is_active = False
    elif args.activate:
        is_active = True

    existing = get_user_by_username(username)
    if existing:
        payload = {"role": role, "is_active": is_active}
        if password:
            payload["password"] = password
        updated = update_user_db(int(existing["id"]), payload)
        print(f"UPDATED_ID={updated['id']}")
        print(f"USERNAME={updated['username']}")
        print(f"ROLE={updated['role']}")
        print(f"ACTIVE={updated['is_active']}")
        print(f"UPDATED_AT={datetime.now().isoformat()}")
        return 0
    else:
        created = create_user_db(username, password, role=role, is_active=is_active)
        print(f"CREATED_ID={created['id']}")
        print(f"USERNAME={created['username']}")
        print(f"ROLE={created['role']}")
        print(f"ACTIVE={created['is_active']}")
        print(f"CREATED_AT={created['created_at']}")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERR:{e}", file=sys.stderr)
        sys.exit(1)
