# page/admin_dashboard.py
# Mirror of pages/admin_dashboard.py for backwards compatibility
import os
import sys

pages_dir = os.path.join(os.path.dirname(__file__), "..", "pages")
sys.path.insert(0, pages_dir)

# Execute pages/admin_dashboard.py
with open(os.path.join(pages_dir, "admin_dashboard.py"), "r", encoding="utf-8") as f:
    code = f.read()
exec(code)
