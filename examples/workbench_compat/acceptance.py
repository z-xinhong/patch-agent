import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from orders import export_orders

assert list(csv.reader(io.StringIO(export_orders([{"id": "A-1"}, {}, {"id": "B,2"}])))) == [
    ["order_ref"], ["A-1"], ["--"], ["B,2"],
]
print("Legacy CSV contract passed.")
