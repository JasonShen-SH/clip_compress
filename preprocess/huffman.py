import heapq
from collections import defaultdict, Counter
import random
import pdb
import torch

# tmp = torch.load("tmp.pt", map_location="cuda")
# indices = [int(element.item()) for tensor in tmp for element in tensor]

# pdb.set_trace()

# 构建哈夫曼树
def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# 对整数索引进行哈夫曼编码
def huffman_encoding(indices):
    frequencies = Counter(indices)
    huffman_tree = build_huffman_tree(frequencies)
    huff_dict = {symbol: code for symbol, code in huffman_tree}
    encoded_data = ''.join(huff_dict[symbol] for symbol in indices)
    return encoded_data, huff_dict

def huff_encode_validate(input_data, huff_dict):
    encoded_data = ''.join(huff_dict[symbol] for symbol in input_data)


# 对哈夫曼编码后的数据进行解码
def huffman_decoding(encoded_data, huff_dict):
    reversed_dict = {code: symbol for symbol, code in huff_dict.items()}
    symbol = ''
    decoded_indices = []
    for bit in encoded_data:
        symbol += bit
        if symbol in reversed_dict:
            decoded_indices.append(reversed_dict[symbol])
            symbol = ''
    return decoded_indices
    

# encoded_data, huff_dict = huffman_encoding(indices)

# decoded_indices = huffman_decoding(encoded_data, huff_dict)