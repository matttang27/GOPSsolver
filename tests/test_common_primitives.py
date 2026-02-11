from __future__ import annotations

import unittest

from tests import test_support

from common import (
    State,
    canonicalize,
    canonicalize_points,
    compress_cards,
    decode_key,
    encode_key,
    guaranteed,
    highest_card,
    list_cards,
    list_to_mask,
    lowest_card,
    popcount,
    remove_card,
    state_from_key,
    state_to_key,
)


class TestCommonPrimitives(unittest.TestCase):
    def test_encode_decode_roundtrip_edge_values(self) -> None:
        cases = [
            (0, 0, 0, 0, 0),
            (0xFFFF, 0x1234, 0x00FF, 127, 255),
            (0x0001, 0x8000, 0x0F0F, -128, 1),
            (0xAAAA, 0x5555, 0x3333, -1, 42),
        ]
        for A, B, P, diff, curP in cases:
            with self.subTest(case=(A, B, P, diff, curP)):
                key = encode_key(A, B, P, diff, curP)
                self.assertEqual(decode_key(key), (A & 0xFFFF, B & 0xFFFF, P & 0xFFFF, diff, curP & 0xFF))

    def test_state_roundtrip(self) -> None:
        state = State(A=0x1234, B=0x00FF, P=0x3333, diff=-7, curP=9)
        key = state_to_key(state)
        recovered = state_from_key(key)
        self.assertEqual(recovered, state)

    def test_list_mask_roundtrip(self) -> None:
        cards = [1, 3, 5, 9]
        m = list_to_mask(cards)
        self.assertEqual(list_cards(m), cards)

    def test_remove_card_popcount_lowest_card(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (test_support.mask([1]), 1, 1, 1),
            (test_support.mask([4]), 1, 4, 4),
            (test_support.mask([13]), 1, 13, 13),
            (test_support.mask([2, 4, 5]), 3, 2, 5),
            (test_support.mask([1, 7, 13]), 3, 1, 13),
            (test_support.mask([3, 4, 5, 6]), 4, 3, 6),
        ]

        for mask, expected_count, expected_lowest, expected_highest in cases:
            with self.subTest(mask=mask):
                self.assertEqual(popcount(mask), expected_count)
                self.assertEqual(lowest_card(mask), expected_lowest)
                self.assertEqual(highest_card(mask), expected_highest)

        m = test_support.mask([2, 4, 5])
        self.assertEqual(popcount(remove_card(m, 4)), 2)
        self.assertEqual(lowest_card(remove_card(m, 2)), 4)
        self.assertEqual(highest_card(remove_card(m, 5)), 4)
        self.assertEqual(popcount(remove_card(0, 9)), 0)
        self.assertEqual(lowest_card(remove_card(0, 9)), 0)
        self.assertEqual(highest_card(remove_card(0, 9)), 0)

    def test_canonicalize_negative_diff_swaps_players(self) -> None:
        A = test_support.mask([1, 3])
        B = test_support.mask([2, 4])
        key1, sign1 = canonicalize(A, B, 0, -2, 5)
        key2, sign2 = canonicalize(B, A, 0, 2, 5)
        self.assertEqual(key1, key2)
        self.assertEqual(sign1, -1.0)
        self.assertEqual(sign2, 1.0)

    def test_canonicalize_points_swaps_by_hand_order(self) -> None:
        A = test_support.mask([1, 2])
        B = test_support.mask([1, 3])
        key1, sign1 = canonicalize_points(A, B, 0, 4)
        key2, sign2 = canonicalize_points(B, A, 0, 4)
        self.assertEqual(key1, key2)
        self.assertEqual(sign1, -1.0)
        self.assertEqual(sign2, 1.0)

    def test_compress_cards_basic_properties(self) -> None:
        a = test_support.mask([2, 8, 12])
        b = test_support.mask([3, 9, 15])
        comp_a, comp_b = compress_cards(a, b)
        self.assertEqual(list_cards(comp_a), [1, 3, 5])
        self.assertEqual(list_cards(comp_b), [2, 4, 6])

    def test_guaranteed_edge_cases(self) -> None:
        self.assertEqual(guaranteed((4, 5), (1, 2), 1, (1, 2)), 1)
        self.assertEqual(guaranteed((1, 2), (4, 5), -1, (1, 2)), -1)
        self.assertEqual(guaranteed((1, 2, 3), (1, 2, 3), 0, (1, 2, 3)), 0)


if __name__ == "__main__":
    unittest.main()
