def main() -> None:
    for row in range(160):
        print(f"adapter setup {row:03} profile=default stream=orders")
    print("LEGACY EXPORT CONTRACT: ID column is order_ref and a missing order ID is rendered as --.")
    for row in range(160):
        print(f"adapter cleanup {row:03} profile=default stream=orders")


if __name__ == "__main__":
    main()
