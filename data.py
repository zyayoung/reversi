import os

def read_data(root):
    records = []
    for path in os.listdir(root):
        with open(os.path.join(root, path), 'rb') as f:
            f.read(4)
            n1 = int.from_bytes(f.read(4), 'little')
            n2 = int.from_bytes(f.read(2), 'little')
            year = int.from_bytes(f.read(2), 'little')
            size = int.from_bytes(f.read(1), 'little')
            assert size == 8
            f.read(3)
            while f.read(8):
                record = []
                for b in f.read(60):
                    if b:
                        r, c = b // 10 - 1, b % 10 - 1
                        assert 0 <= r < 8, (len(record), r, c)
                        assert 0 <= c < 8, (len(record), r, c)
                        record.append(r*8+c)
                records.append(record)
    return records

if __name__ == "__main__":
    data = read_data("data")
    print(len(data))
