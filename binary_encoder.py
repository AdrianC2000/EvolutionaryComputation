import math


class BinaryEncoder:

    def __init__(self, precision: int, limit_start: int, limit_end: int) -> None:
        self.__limit_start = limit_start
        self.__limit_range = limit_end - limit_start
        self.__binary_chain_length = self._calculate_binary_chain_length(precision)

    def _calculate_binary_chain_length(self, precision: int) -> int:
        return math.ceil(math.log2(self.__limit_range * 10 ** precision + 1))

    def encode_to_binary(self, number: float) -> str:
        fractional_part = (number - self.__limit_start) * (2 ** self.__binary_chain_length - 1) / self.__limit_range
        fractional_int = int(fractional_part)
        binary_fractional_part = format(fractional_int, 'b')
        binary_fractional_part = binary_fractional_part.zfill(self.__binary_chain_length)
        return binary_fractional_part

    def decode_from_binary(self, binary: str) -> float:
        decimal = int(binary, 2)
        decoded_number = self.__limit_start + decimal * self.__limit_range / (2 ** self.__binary_chain_length - 1)
        return decoded_number