"""Utility functions for GOPS solver"""

from functools import lru_cache
from typing import Union

def cmp(a: Union[int, float], b: Union[int, float]) -> int:
    """Compare two numbers and return -1, 0, or 1 (spaceship operator)."""
    return (a > b) - (a < b)

@lru_cache(maxsize=None)
def compress_cards(cardsA: tuple[int, ...], cardsB: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Compress card values to remove gaps while preserving relative order.
    """
    # Get all unique values and sort them
    all_values = sorted(set(cardsA) | set(cardsB))
    
    # Create mapping from old values to new compressed values
    value_map = {old_val: new_val for new_val, old_val in enumerate(all_values, 1)}
    
    # Apply mapping to both card sets
    compressed_A = tuple(value_map[card] for card in cardsA)
    compressed_B = tuple(value_map[card] for card in cardsB)
    
    return compressed_A, compressed_B

def guaranteed(cardsA: tuple[int, ...], cardsB: tuple[int, ...], pointDiff: int, prizes: tuple[int, ...]) -> int:
    """
    Check if one side has enough cards higher than the other to guarantee a win.
    """
    cardsLeft = len(prizes)
    sorted_prizes = sorted(prizes, reverse=True)
    guarantee = [sum(sorted_prizes[:i]) - sum(sorted_prizes[i:]) for i in range(cardsLeft + 1)]
    
    guaranteeA = sum(1 for card in cardsA if card > cardsB[-1])
    guaranteeB = sum(1 for card in cardsB if card > cardsA[-1])
    
    if (guarantee[guaranteeA] + pointDiff) > 0:
        return 1
    elif (pointDiff - guarantee[guaranteeB]) < 0:
        return -1
    else:
        return 0

def full(n):
    """Returns a tuple from 1 to n"""
    return tuple(i for i in range(1, n + 1))