# Strong Typing in Python - Complete Guide

## Overview
Python is dynamically typed by default, but you can add strong typing using:
1. **Type hints** (annotations) - for static analysis
2. **Static type checkers** (mypy, pyright) - catch errors before runtime
3. **Runtime type checking** (optional) - validate types during execution

## 1. Type Hints Syntax

### Basic Types
```python
from typing import List, Dict, Optional, Union, Tuple, Any

def basic_types(
    name: str,
    age: int,
    height: float,
    is_student: bool
) -> str:
    return f"{name} is {age} years old"
```

### Collections
```python
def collections_example(
    numbers: list[int],              # Python 3.9+
    scores: dict[str, float],        # Python 3.9+
    coordinates: tuple[int, int],    # Python 3.9+
    names: List[str]                 # Pre-3.9 (requires import)
) -> None:
    pass
```

### Optional and Union Types
```python
def optional_example(
    value: Optional[int] = None,     # Same as Union[int, None]
    result: Union[str, int] = "default"
) -> Optional[str]:
    if value is None:
        return None
    return str(value)
```

### Function Types
```python
from typing import Callable

def higher_order(
    func: Callable[[int, int], int],  # Function taking 2 ints, returning int
    x: int,
    y: int
) -> int:
    return func(x, y)
```

## 2. Advanced Type Hints

### Generic Types
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
```

### Protocol (Structural Typing)
```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()  # Any object with a draw() method works
```

### Literal Types
```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass

set_mode("read")     # ✅ Valid
set_mode("delete")   # ❌ Type error
```

## 3. Class Type Hints

```python
from __future__ import annotations  # Enables forward references

class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
        self.friends: list[Person] = []  # Forward reference works with __future__
    
    def add_friend(self, friend: Person) -> None:
        self.friends.append(friend)
    
    @classmethod
    def from_string(cls, data: str) -> Person:
        name, age_str = data.split(',')
        return cls(name, int(age_str))
```

## 4. Type Checking Tools

### mypy (Static Type Checker)
```bash
# Install
pip install mypy

# Basic usage
mypy your_file.py

# With configuration
mypy --config-file mypy.ini your_file.py
```

### Runtime Type Checking
```python
# Option 1: typeguard
from typeguard import typechecked

@typechecked
def safe_divide(a: int, b: int) -> float:
    return a / b

# Option 2: pydantic for data validation
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str
    
user = User(name="John", age=30, email="john@example.com")
```

## 5. Best Practices

### 1. Start with Return Types
```python
def calculate_total(items: list[float]) -> float:  # Clear return type
    return sum(items)
```

### 2. Use Optional for Nullable Values
```python
def find_user(user_id: int) -> Optional[User]:
    # Might return None
    return database.get_user(user_id)
```

### 3. Prefer Union Over Any
```python
# Good
def process_id(user_id: Union[str, int]) -> str:
    return str(user_id)

# Avoid (loses type safety)
def process_id(user_id: Any) -> str:
    return str(user_id)
```

### 4. Use TypedDict for Structured Dictionaries
```python
from typing import TypedDict

class PersonDict(TypedDict):
    name: str
    age: int
    email: str

def process_person(person: PersonDict) -> str:
    return f"{person['name']} ({person['age']})"
```

## 6. Configuration Files

### mypy.ini
```ini
[mypy]
python_version = 3.9
strict_optional = True
warn_return_any = True
show_error_codes = True

[mypy-external_lib.*]
ignore_missing_imports = True
```

### pyproject.toml (modern approach)
```toml
[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true

[[tool.mypy.overrides]]
module = "external_lib.*"
ignore_missing_imports = true
```

## 7. IDE Integration

Most modern IDEs (VS Code, PyCharm, etc.) support type hints:
- **Autocomplete** - Better suggestions based on types
- **Error detection** - Catch type errors while coding
- **Refactoring** - Safer code transformations
- **Documentation** - Types serve as inline documentation

## 8. Gradual Typing Strategy

1. **Start small** - Add types to new functions first
2. **Focus on public APIs** - Type public interfaces first
3. **Use mypy incrementally** - Enable strict checking gradually
4. **Document complex types** - Add docstrings for complex type relationships

## Example: Complete Typed Module

```python
from __future__ import annotations
from typing import Optional, Union, Protocol
from dataclasses import dataclass

class Comparable(Protocol):
    def __lt__(self, other: Comparable) -> bool: ...

@dataclass
class GameResult:
    player_a_score: int
    player_b_score: int
    winner: Optional[str] = None

def compare_values(a: Union[int, float], b: Union[int, float]) -> int:
    """Spaceship operator: returns -1, 0, or 1."""
    return (a > b) - (a < b)

def determine_winner(result: GameResult) -> str:
    """Determine the winner based on scores."""
    comparison = compare_values(result.player_a_score, result.player_b_score)
    
    if comparison > 0:
        return "Player A"
    elif comparison < 0:
        return "Player B"
    else:
        return "Tie"
```

This gives you comprehensive type safety while maintaining Python's flexibility!
