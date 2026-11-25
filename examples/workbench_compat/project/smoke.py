from orders import list_orders

assert list_orders([{"id": "A-1"}]) == [{"id": "A-1"}]
print("Order listing smoke check passed.")
